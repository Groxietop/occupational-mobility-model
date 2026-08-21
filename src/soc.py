"""SOC code plumbing.

Three code systems have to line up before anything else works:

  O*NET      10-digit detailed codes   '15-1252.00', '15-1252.01'
  SOC        6-digit occupation codes  '15-1252'      (7 chars with the hyphen)
  Census OCC 4-digit CPS codes         '1021'

CPS reports occupation in Census OCC codes, O*NET publishes features against
its own detailed codes, and the crosswalk joins them at the SOC-6 level. So
SOC-6 is the common currency: everything gets collapsed to it before any
pair-level work happens.
"""

from __future__ import annotations

import pandas as pd

# O*NET's `required_education_level` is a categorical string. Ordering it lets
# us measure how far a move reaches across an education boundary, which is one
# of the strongest non-skill barriers to occupational mobility. The scale is
# ordinal, not interval -- the gaps between levels are not claimed to be equal.
EDUCATION_RANK = {
    "Less than a High School Diploma": 1.0,
    "High School Diploma - or the equivalent (for example, GED)": 2.0,
    "Post-Secondary Certificate": 3.0,
    "Some College Courses": 3.5,
    "Associate's Degree (or other 2-year degree)": 4.0,
    "Bachelor's Degree": 5.0,
    "Post-Baccalaureate Certificate": 5.5,
    "Master's Degree": 6.0,
    "Post-Master's Certificate": 6.5,
    "First Professional Degree": 7.0,
    "Doctoral Degree": 7.0,
    "Post-Doctoral Training": 8.0,
}


def to_soc6(onet_code: str) -> str:
    """'15-1252.00' -> '15-1252'. Already-6-digit codes pass through."""
    return str(onet_code).strip()[:7]


def rank_education(label) -> float | None:
    """Map an O*NET education label to its ordinal rank.

    O*NET spells several levels out with a long parenthetical definition, so
    match on the leading phrase rather than the full string.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    text = str(label).strip()
    if text in EDUCATION_RANK:
        return EDUCATION_RANK[text]
    for prefix, rank in EDUCATION_RANK.items():
        # The long-form labels start with the short label then " - awarded for..."
        key = prefix.split(" - ")[0]
        if text.startswith(key):
            return rank
    return None


def collapse_to_soc6(master: pd.DataFrame) -> pd.DataFrame:
    """Collapse the O*NET master from detailed codes to one row per SOC-6.

    Several O*NET detail codes share a SOC-6 (15-1252.00 and 15-1252.01 are
    both '15-1252'), but CPS only ever resolves to SOC-6. Numeric feature
    columns are averaged across the detail codes; the title of the first
    (`.00`, the base occupation, when present) is kept for readability.
    """
    feature_cols = [c for c in master.columns if "__" in c]
    soc6 = master["soc_code"].map(to_soc6)

    # Assemble the frame to be grouped in one construction. The master is ~445
    # columns wide, and building it by inserting columns leaves pandas with a
    # fragmented frame that it warns about on every groupby.
    numeric = pd.concat(
        [
            master[feature_cols + ["job_zone"]],
            master["required_education_level"].map(rank_education).rename(
                "education_rank"
            ),
        ],
        axis=1,
    ).copy()
    numeric.insert(0, "soc6", soc6)

    agg = numeric.groupby("soc6", as_index=False).mean()

    # Prefer the base '.00' code's title; fall back to whichever comes first.
    titles = (
        pd.DataFrame({"soc6": soc6, "soc_code": master["soc_code"], "title": master["title"]})
        .sort_values("soc_code")
        .groupby("soc6", as_index=False)
        .first()[["soc6", "title"]]
    )

    out = agg.merge(titles, on="soc6", how="left")
    cols = ["soc6", "title", "job_zone", "education_rank"] + feature_cols
    return out[cols]


def load_crosswalk(path) -> pd.DataFrame:
    """Census OCC (4-digit, zero-padded) -> SOC-6."""
    xwalk = pd.read_csv(path, dtype=str)
    xwalk["census_occ_code"] = xwalk["census_occ_code"].str.strip().str.zfill(4)
    xwalk["soc6"] = xwalk["soc_code_6digit"].str.strip().str[:7]
    return xwalk[["census_occ_code", "census_title", "soc6"]]

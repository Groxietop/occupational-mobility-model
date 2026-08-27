"""Build `onet_master.parquet` from an O*NET bulk database.

    python src/build_master.py --raw data/raw/onet_31_0/db_31_0_text
    python src/build_master.py --raw <dir> --validate-against data/processed/onet_master.parquet

Nothing in this repo used to build the master. Every module read it, and
the file itself was the only copy of how it had been made -- which meant a
new O*NET release could not be picked up without reverse-engineering the
artifact first. That is what this script exists to end.

Column shape, matching what the original master had:

    {domain}__{Element Name}__{Scale ID}    most domains
    work_context__{Element Name}            context only, CX scale

Scales kept per domain, chosen to match the existing file exactly:

    ability, skill, knowledge, work_activity   IM (importance), LV (level)
    work_style                                 DR, WI
    work_value                                 EX, VH
    interest                                   OI, IH
    work_context                               CX only, no scale suffix

`--validate-against` diffs a freshly built frame against an existing master.
Run it on the version the old file came from before trusting the builder on
a new one: if it cannot reproduce 30.2, it should not be believed about 31.0.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# (source files, column prefix, scales to keep, whether to suffix the scale)
#
# O*NET 31.0 reorganised the content model. Several domains moved files
# without changing content, and one was retired outright:
#
#   Skills.txt      -> Essential Skills.txt + Transferable Skills.txt
#                      (same 35 elements, just split by category)
#   Interests.txt   -> Career Interest Types.txt
#                      (same 9 elements)
#   Work Values.txt -> REMOVED. Not relocated -- work values are gone from
#                      the 31.0 Content Model Reference entirely.
#   Technology Skills.txt -> Software Skills.txt
#   Education, Training, and Experience.txt -> Education.txt + Training...
#
# Listing several candidate filenames per domain lets one builder handle
# both layouts, and missing files are skipped rather than fatal -- that is
# how a retired domain simply drops out instead of breaking the build.
DOMAIN_SPECS = [
    (["Abilities.txt"], "ability", ("IM", "LV"), True),
    (["Skills.txt", "Essential Skills.txt", "Transferable Skills.txt"],
     "skill", ("IM", "LV"), True),
    (["Knowledge.txt"], "knowledge", ("IM", "LV"), True),
    (["Work Activities.txt"], "work_activity", ("IM", "LV"), True),
    (["Work Styles.txt"], "work_style", ("DR", "WI"), True),
    (["Work Values.txt"], "work_value", ("EX", "VH"), True),
    (["Interests.txt", "Career Interest Types.txt"], "interest", ("OI", "IH"), True),
    (["Work Context.txt"], "work_context", ("CX",), False),
]

SOC = "O*NET-SOC Code"
ELEMENT = "Element Name"
SCALE = "Scale ID"
VALUE = "Data Value"


def _read(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point --raw at the db_XX_X_text directory "
            "inside an O*NET bulk download."
        )
    return pd.read_csv(path, sep="\t", dtype=str, quoting=3)


def _domain_frame(
    raw_dir: Path, filenames, prefix: str, scales, suffix_scale: bool
) -> pd.DataFrame | None:
    """Build one domain's wide frame, concatenating any files that exist.

    Returns None when no candidate file is present -- a domain O*NET has
    retired should vanish from the feature space, not crash the build or
    silently carry stale values from an older release.
    """
    present = [f for f in filenames if (raw_dir / f).exists()]
    if not present:
        return None
    df = pd.concat([_read(raw_dir, f) for f in present], ignore_index=True)
    df = df[df[SCALE].isin(scales)].copy()
    df[VALUE] = pd.to_numeric(df[VALUE], errors="coerce")

    if suffix_scale:
        df["__col"] = prefix + "__" + df[ELEMENT] + "__" + df[SCALE]
    else:
        df["__col"] = prefix + "__" + df[ELEMENT]

    # Some elements appear more than once per (occupation, scale) across
    # domain sources; take the mean rather than letting pivot raise.
    wide = df.pivot_table(index=SOC, columns="__col", values=VALUE, aggfunc="mean")
    wide.columns.name = None
    return wide


def _occupations(raw_dir: Path) -> pd.DataFrame:
    occ = _read(raw_dir, "Occupation Data.txt")
    return occ.rename(
        columns={SOC: "soc_code", "Title": "title", "Description": "description"}
    )[["soc_code", "title", "description"]]


def _job_zones(raw_dir: Path) -> pd.DataFrame:
    jz = _read(raw_dir, "Job Zones.txt")
    jz = jz.rename(columns={SOC: "soc_code", "Job Zone": "job_zone"})
    jz["job_zone"] = pd.to_numeric(jz["job_zone"], errors="coerce")
    return jz.groupby("soc_code", as_index=False)["job_zone"].first()


def _education(raw_dir: Path) -> pd.DataFrame:
    """Modal required education category per occupation.

    The Education/Training/Experience table gives a percentage per category;
    the master carried a single label, so take the highest-percentage one.
    """
    if (raw_dir / "Education.txt").exists():        # 31.0 layout
        ete = _read(raw_dir, "Education.txt")
        cats = _read(raw_dir, "Education Categories.txt")
    else:                                            # 30.2 and earlier
        ete = _read(raw_dir, "Education, Training, and Experience.txt")
        cats = _read(raw_dir, "Education, Training, and Experience Categories.txt")

    required = ete[ete[SCALE] == "RL"].copy()
    if required.empty:
        return pd.DataFrame(columns=["soc_code", "required_education_level"])
    required[VALUE] = pd.to_numeric(required[VALUE], errors="coerce")

    cat_lookup = cats[cats[SCALE] == "RL"][["Category", "Category Description"]]
    cat_lookup = dict(
        zip(cat_lookup["Category"].astype(str), cat_lookup["Category Description"])
    )

    top = required.sort_values(VALUE, ascending=False).groupby(SOC, as_index=False).first()
    top["required_education_level"] = top["Category"].astype(str).map(cat_lookup)
    return top.rename(columns={SOC: "soc_code"})[
        ["soc_code", "required_education_level"]
    ]


def _joined_list(raw_dir: Path, filename: str, column: str, out_name: str) -> pd.DataFrame:
    """Comma-joined unique values per occupation, e.g. technology skills."""
    df = _read(raw_dir, filename)
    if column not in df.columns:
        return pd.DataFrame(columns=["soc_code", out_name])
    grouped = (
        df.groupby(SOC)[column]
        .apply(lambda s: ", ".join(sorted(set(s.dropna()))))
        .reset_index()
    )
    return grouped.rename(columns={SOC: "soc_code", column: out_name})


def _technology_file(raw_dir: Path) -> str | None:
    for name in ("Technology Skills.txt", "Software Skills.txt"):
        if (raw_dir / name).exists():
            return name
    return None


def _technology(raw_dir: Path) -> pd.DataFrame:
    """Technology categories per occupation.

    30.2 calls the category column 'Commodity Title'; 31.0's Software Skills
    calls it 'Element Name'.
    """
    filename = _technology_file(raw_dir)
    if filename is None:
        return pd.DataFrame(columns=["soc_code", "technology_skills"])
    df = _read(raw_dir, filename)
    column = "Commodity Title" if "Commodity Title" in df.columns else "Element Name"
    grouped = (
        df.groupby(SOC)[column]
        .apply(lambda s: ", ".join(sorted(set(s.dropna()))))
        .reset_index()
    )
    return grouped.rename(columns={SOC: "soc_code", column: "technology_skills"})


def _hot_technologies(raw_dir: Path) -> pd.DataFrame:
    filename = _technology_file(raw_dir)
    if filename is None:
        return pd.DataFrame(columns=["soc_code", "hot_technologies"])
    df = _read(raw_dir, filename)
    if "Hot Technology" not in df.columns:
        return pd.DataFrame(columns=["soc_code", "hot_technologies"])
    hot = df[df["Hot Technology"].astype(str).str.upper() == "Y"]
    example = "Example" if "Example" in df.columns else "Workplace Example"
    grouped = (
        hot.groupby(SOC)[example]
        .apply(lambda s: ", ".join(sorted(set(s.dropna()))))
        .reset_index()
    )
    return grouped.rename(columns={SOC: "soc_code", example: "hot_technologies"})


def _related(raw_dir: Path) -> pd.DataFrame:
    """O*NET's own related occupations, primary ones only."""
    df = _read(raw_dir, "Related Occupations.txt")
    code_col = next(
        (c for c in df.columns if "Related" in c and "Code" in c), None
    )
    if code_col is None:
        return pd.DataFrame(columns=["soc_code", "primary_related_soc_codes"])

    if "Relatedness Tier" in df.columns:
        df = df[df["Relatedness Tier"].str.contains("Primary", na=False)]

    grouped = (
        df.groupby(SOC)[code_col]
        .apply(lambda s: ", ".join(s.dropna().astype(str)))
        .reset_index()
    )
    return grouped.rename(columns={SOC: "soc_code", code_col: "primary_related_soc_codes"})


DIR_VERSION = re.compile(r"db_(\d+)_(\d+)_text")


def source_version(raw_dir: Path | str) -> str | None:
    """Version of a bulk directory, e.g. .../db_31_0_text -> '31.0'."""
    match = DIR_VERSION.search(str(raw_dir))
    return f"{match.group(1)}.{match.group(2)}" if match else None


def build(raw_dir: Path | str) -> pd.DataFrame:
    raw_dir = Path(raw_dir)

    master = _occupations(raw_dir)
    for frame in (
        _job_zones(raw_dir),
        _education(raw_dir),
        _technology(raw_dir),
        _hot_technologies(raw_dir),
        _related(raw_dir),
    ):
        master = master.merge(frame, on="soc_code", how="left")

    domain_frames = []
    for filenames, prefix, scales, suffix in DOMAIN_SPECS:
        frame = _domain_frame(raw_dir, filenames, prefix, scales, suffix)
        if frame is None:
            print(f"  [note] no source file for '{prefix}' — domain absent in this release")
            continue
        domain_frames.append(frame)
    features = pd.concat(domain_frames, axis=1)
    features.index.name = "soc_code"

    master = master.merge(features.reset_index(), on="soc_code", how="left")
    master = master.sort_values("soc_code").reset_index(drop=True)

    # Stamp the source release into the artifact. The freshness check used to
    # infer it from a directory name in data/raw/, which breaks the moment the
    # raw dump isn't committed -- it then reports whatever stale folder is
    # lying around rather than what the master was actually built from.
    master.attrs["onet_version"] = source_version(raw_dir)
    if "onet_version" not in master.columns:
        master.insert(1, "onet_version", source_version(raw_dir))
    return master


def validate(built: pd.DataFrame, existing_path: Path | str) -> dict:
    """Diff a freshly built master against an existing one."""
    existing = pd.read_parquet(existing_path).sort_values("soc_code").reset_index(drop=True)

    built_cols, existing_cols = set(built.columns), set(existing.columns)
    shared = sorted(built_cols & existing_cols)

    report = {
        "rows_built": len(built),
        "rows_existing": len(existing),
        "columns_built": len(built_cols),
        "columns_existing": len(existing_cols),
        "missing_from_build": sorted(existing_cols - built_cols),
        "extra_in_build": sorted(built_cols - existing_cols),
        "same_occupations": set(built["soc_code"]) == set(existing["soc_code"]),
    }

    numeric = [
        c for c in shared
        if pd.api.types.is_numeric_dtype(existing[c]) and c != "soc_code"
    ]
    merged = built[["soc_code"] + numeric].merge(
        existing[["soc_code"] + numeric], on="soc_code", suffixes=("_new", "_old")
    )
    mismatched = []
    for col in numeric:
        a, b = merged[f"{col}_new"], merged[f"{col}_old"]
        differs = ~((a - b).abs() <= 0.005) & ~(a.isna() & b.isna())
        if differs.any():
            mismatched.append((col, int(differs.sum())))
    report["numeric_columns_checked"] = len(numeric)
    report["numeric_columns_mismatched"] = mismatched
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Path to a db_XX_X_text directory")
    parser.add_argument("--out", default=None, help="Where to write the parquet")
    parser.add_argument(
        "--validate-against",
        default=None,
        help="Existing master to diff against instead of writing",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    master = build(args.raw)
    print(f"Built master: {master.shape[0]:,} occupations × {master.shape[1]:,} columns")

    if args.validate_against:
        report = validate(master, args.validate_against)
        print(f"\nValidating against {args.validate_against}")
        print(f"  rows      built {report['rows_built']:,} vs existing {report['rows_existing']:,}")
        print(f"  columns   built {report['columns_built']:,} vs existing {report['columns_existing']:,}")
        print(f"  same occupation set: {report['same_occupations']}")
        if report["missing_from_build"]:
            print(f"  MISSING from build ({len(report['missing_from_build'])}): "
                  f"{report['missing_from_build'][:6]}")
        if report["extra_in_build"]:
            print(f"  EXTRA in build ({len(report['extra_in_build'])}): "
                  f"{report['extra_in_build'][:6]}")
        print(f"  numeric columns checked: {report['numeric_columns_checked']:,}")
        if report["numeric_columns_mismatched"]:
            print(f"  MISMATCHED ({len(report['numeric_columns_mismatched'])}): "
                  f"{report['numeric_columns_mismatched'][:6]}")
        else:
            print("  every shared numeric column matches")
        return 0

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        master.to_parquet(out, index=False)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""O*NET Web Services (v2), for keeping the local database honest.

The other two sources in this project fetch their data over an API. O*NET
does not, and that is deliberate rather than unfinished.

Why the bulk download stays
---------------------------
The v2 API serves the ratings the similarity model needs, but paginated ten
elements at a time. One occupation's skills alone is four requests; the full
matrix is 1,016 occupations x 8 domains x ~4 pages, on the order of **32,000
requests** for a single rebuild. The bulk database is one download of the
same numbers. Rebuilding the feature matrix over the API would be slower,
more fragile, and no more current.

What the API is genuinely good for is the thing a static dump can never do:
telling you it has gone stale. `data/raw/onet_30_2/` is version 30.2. At the
time of writing the service reports **31.0** — two releases behind, and
nothing in the repo would have said so.

So this module answers one question well: *is the local database still the
current one, and what changed if not?* Plus spot lookups for a single
occupation, which is cheap and useful when checking a specific case by hand.

Auth
----
`X-API-Key` header against `https://api-v2.onetcenter.org`. Note the legacy
`services.onetcenter.org/ws/` endpoints use a different, older
username/password scheme and reject these keys with a 401 — if you get one,
check the host before assuming the key is wrong.

Key comes from `ONET_API_KEY`. Sign up at
https://services.onetcenter.org/developer/signup.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API_ROOT = "https://api-v2.onetcenter.org"

# O*NET asks API consumers to identify themselves.
USER_AGENT = "occupational-mobility-model/0.1 (+https://github.com/Groxietop/occupational-mobility-model)"

# The eight rating domains the similarity model is built from, mapped to
# their v2 detail paths.
DOMAIN_PATHS = {
    "ability": "abilities",
    "skill": "skills",
    "knowledge": "knowledge",
    "work_activity": "work_activities",
    "work_style": "work_styles",
    "work_value": "work_values",
    "interest": "interests",
    "work_context": "work_context",
}

# Local bulk database, e.g. data/raw/onet_30_2 -> "30.2"
LOCAL_DIR_PATTERN = re.compile(r"onet_(\d+)_(\d+)")


class ONetError(RuntimeError):
    pass


def get_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("ONET_API_KEY")
    if not key:
        raise ONetError(
            "ONET_API_KEY is not set. Sign up at "
            "https://services.onetcenter.org/developer/signup and export it, "
            "or put it in a local .env file (which is gitignored)."
        )
    return key


def _get(path: str, api_key: str | None = None, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(
        f"{API_ROOT}/{path.lstrip('/')}",
        headers={
            "X-API-Key": get_key(api_key),
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ONetError(
                "401 from O*NET. If you are pointing at "
                "services.onetcenter.org/ws/, that is the legacy API and uses "
                "username/password; v2 keys only work against "
                f"{API_ROOT}."
            ) from exc
        raise ONetError(f"O*NET returned {exc.code} for {path}") from exc


# --- version / freshness ---------------------------------------------------


@dataclass(frozen=True)
class Freshness:
    """Whether the committed bulk database is still the current release."""

    local_version: str | None
    published_version: str | None
    taxonomy: str | None = None

    @property
    def known(self) -> bool:
        return bool(self.local_version and self.published_version)

    @property
    def is_current(self) -> bool:
        return self.known and self.local_version == self.published_version

    def describe(self) -> str:
        if not self.known:
            return (
                f"could not compare versions "
                f"(local={self.local_version}, published={self.published_version})"
            )
        if self.is_current:
            return f"O*NET {self.local_version} is current"
        return (
            f"local O*NET database is {self.local_version}; O*NET now publishes "
            f"{self.published_version}. Re-download the bulk database from "
            "https://www.onetcenter.org/database.html and rebuild the master."
        )


def published_version(api_key: str | None = None) -> tuple[str | None, str | None]:
    """(database version, taxonomy name) the service is serving right now."""
    payload = _get("about", api_key=api_key)
    database = (payload.get("database") or {}).get("name") or ""
    taxonomy = (payload.get("taxonomy") or {}).get("name")
    # "O*NET 31.0" -> "31.0"
    match = re.search(r"(\d+\.\d+)", database)
    return (match.group(1) if match else None), taxonomy


def version_from_master(master_path: Path | str) -> str | None:
    """Release the master was built from, as stamped by build_master.py.

    This is the authoritative answer. Reading a directory name under
    data/raw/ only tells you which bulk dump happens to be sitting on disk,
    which is not the same thing -- and is wrong outright once the raw dump
    stops being committed.
    """
    master_path = Path(master_path)
    if not master_path.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_parquet(master_path, columns=["onet_version"])
    except Exception:
        return None
    if frame.empty:
        return None
    value = frame["onet_version"].dropna()
    return str(value.iloc[0]) if len(value) else None


def local_version(
    raw_dir: Path | str, master_path: Path | str | None = None
) -> str | None:
    """Version the pipeline is actually running on.

    Prefers the stamp inside the master; falls back to a bulk directory name
    for masters built before stamping existed.
    """
    if master_path is not None:
        stamped = version_from_master(master_path)
        if stamped:
            return stamped

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    for child in sorted(raw_dir.iterdir()):
        match = LOCAL_DIR_PATTERN.fullmatch(child.name)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    return None


def check_freshness(
    raw_dir: Path | str,
    api_key: str | None = None,
    master_path: Path | str | None = None,
) -> Freshness:
    """Compare what the pipeline runs on against what O*NET publishes today."""
    local = local_version(raw_dir, master_path=master_path)
    try:
        published, taxonomy = published_version(api_key=api_key)
    except ONetError:
        return Freshness(local_version=local, published_version=None)
    return Freshness(local_version=local, published_version=published, taxonomy=taxonomy)


# --- spot lookups ----------------------------------------------------------


def occupation(soc_code: str, api_key: str | None = None) -> dict:
    """Summary for one O*NET-SOC code, e.g. '15-1252.00'."""
    return _get(f"online/occupations/{soc_code}", api_key=api_key)


def _paged_elements(path: str, api_key: str | None, key: str = "element") -> list[dict]:
    """Walk O*NET's pagination, which returns ten elements at a time."""
    out: list[dict] = []
    url = path
    while url:
        payload = _get(url, api_key=api_key)
        out.extend(payload.get(key, []))
        next_url = payload.get("next")
        if not next_url:
            break
        url = next_url.replace(f"{API_ROOT}/", "")
    return out


def domain_ratings(
    soc_code: str, domain: str, api_key: str | None = None
) -> dict[str, float]:
    """Importance ratings for one domain of one occupation.

    Cheap for a single occupation, ruinous across all 1,016 -- see the module
    docstring. Use it to check a specific case, not to rebuild the matrix.
    """
    if domain not in DOMAIN_PATHS:
        raise ValueError(
            f"unknown domain {domain!r}; expected one of {sorted(DOMAIN_PATHS)}"
        )
    elements = _paged_elements(
        f"online/occupations/{soc_code}/details/{DOMAIN_PATHS[domain]}", api_key
    )
    return {
        element["name"]: float(element["importance"])
        for element in elements
        if "importance" in element
    }


def related_occupations(soc_code: str, api_key: str | None = None) -> list[dict]:
    """O*NET's own curated related occupations — the model's baseline."""
    elements = _paged_elements(
        f"online/occupations/{soc_code}/details/related_occupations",
        api_key,
        key="occupation",
    )
    return [{"code": e["code"], "title": e["title"]} for e in elements if "code" in e]


# --- bulk database download ------------------------------------------------

BULK_URL = "https://www.onetcenter.org/dl_files/database/db_{version}_text.zip"


def bulk_download_url(version: str) -> str:
    """Download URL for a bulk text release, e.g. '31.0' -> db_31_0_text.zip."""
    return BULK_URL.format(version=version.replace(".", "_"))


def download_bulk(version: str, target_dir, timeout: float = 300.0):
    """Fetch and unpack an O*NET bulk text database.

    ~13 MB compressed, ~99 MB unpacked, which is why the raw dump is not
    committed for every release -- `build_master.py` turns it into a 1.2 MB
    parquet and that is what the pipeline actually reads.
    """
    import io
    import zipfile

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = bulk_download_url(version)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target_dir)

    unpacked = sorted(p for p in target_dir.iterdir() if p.is_dir())
    if not unpacked:
        raise ONetError(f"nothing unpacked from {url}")
    return unpacked[-1]

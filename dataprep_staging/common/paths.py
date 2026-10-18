"""Fixed locations of the data-remediation layout (plan section 2.2, user decision D12).

Everything that holds data lives under gitignored directories (``results/``,
``data_fetch_api/``); only code lives under ``dataprep_staging/``.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repository root (this file is <repo>/dataprep_staging/common/paths.py). Override with
#: DATAPREP_REPO.
REPO = Path(os.environ.get("DATAPREP_REPO", Path(__file__).resolve().parents[2])).resolve()

STAGING = REPO / "dataprep_staging"
DATAPREP = REPO / "results" / "dataprep"
PILOTS = DATAPREP / "_pilots"
LEDGER_DIR = DATAPREP / "ledger"
LEDGER_ROWS = LEDGER_DIR / "rows"
LEDGER_FILE = LEDGER_DIR / "data_roles.csv"
PREREG_DIR = DATAPREP / "preregistration"
REPORTS_DIR = DATAPREP / "reports"
L0_NEW = REPO / "data_fetch_api" / "new_measurements"

#: Read-only L0 vendor caches (plan 2.1). Nothing in the staging code writes there.
L0_READONLY = (
    REPO / "AMP_V2" / "data",
    REPO / "chou-shui-data" / "data",
    REPO / "data_fetch_api",
)

#: Forecast origin: last month of pre-origin data (plan 4.2).
ORIGIN = "2022-12"

#: Twin grid used by L3 products (plan 2.1; the task's model grid).
TWIN_GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"


def rel_to_repo(path: str | os.PathLike) -> str:
    """Repository-relative POSIX path when ``path`` is inside the repository, else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def from_repo(path: str | os.PathLike) -> Path:
    """Inverse of :func:`rel_to_repo`: absolute path for a repo-relative or absolute string."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p

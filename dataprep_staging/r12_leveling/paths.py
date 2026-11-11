"""Locations for the round-12 exploratory scripts.

``HYDRO_RESULTS`` is the repository's ``results`` directory holding the data products
(default: ``results`` under the working directory). ``HYDRO_R12_OUT`` is a local output
directory outside the repository; nothing written there is committed.
"""

from __future__ import annotations

import os

RESULTS = os.environ.get("HYDRO_RESULTS", "results")
OUT = os.path.expanduser(os.environ.get("HYDRO_R12_OUT", "~/hydro_scratch_r12"))
DATAPREP = os.path.join(RESULTS, "dataprep", "")
HEADS = os.path.join(DATAPREP, "heads", "L1_clean", "v3", "")
FWD = os.path.join(RESULTS, "twin_forward", "datum_gate_innov.npz")
YEARBOOK = os.path.join(OUT, "yearbook", "out", "heads_monthly_yearbook.parquet")
DDIR = os.path.join("chou-shui-data", "data")
POLY = os.path.join(DDIR, "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.json")


def out(name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)

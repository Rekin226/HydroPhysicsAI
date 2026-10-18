"""WP S4a (InSAR setup): fixed locations and fetch windows for the LiCSAR subsets.

Plan section 3 "WP S4" (decisions D1, D11 = none, D12). Only public URLs and window dates
live here; the fan clip box is computed at run time from the twin grid
(``results/twin_forward/datum_gate_innov.npz``) by ``make_run_config.py`` and stored under
``results/`` (gitignored), so no study-area coordinates are versioned.
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGING = Path(__file__).resolve().parents[1]
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, L0_NEW, REPO  # noqa: E402

WP = "S4a"
FETCH_DATE = "2026-10-12"
FETCH_TAG = f"fetch_{FETCH_DATE}_s4a"

#: Public LiCSAR portal (anonymous HTTPS; HTTP range reads accepted). The plain
#: ``LiCSAR_products/`` tree now serves per-interferogram HTML link pages (LiCSAR web layout
#: since 2025-08); ``LiCSAR_products.public/`` serves the files directly (checked 2026-10-12).
LICSAR_PUB = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public"

#: Frames and fetch windows (inclusive yyyymmdd on both epochs of a pair).
FRAMES = {
    "069A_06694_131402": {
        "role": "primary ascending frame (85 % of the fan box, all 14 ring sites)",
        "start": "20200101", "end": "20260131",
        "l0_dir": L0_NEW / "licsar_069A_06694" / FETCH_TAG,
    },
    "069A_06632_162010": {
        "role": "full-fan ascending frame for asc/desc decomposition from 2022-11",
        "start": "20221101", "end": "20260131",
        "l0_dir": L0_NEW / "licsar_069A_06632" / FETCH_TAG,
    },
    "105D_06628_232108": {
        "role": "full-fan descending frame for asc/desc decomposition from 2022-11",
        "start": "20221101", "end": "20260131",
        "l0_dir": L0_NEW / "licsar_105D_06628" / FETCH_TAG,
    },
}

#: Per-interferogram products LiCSBAS needs (LiCSBAS01_get_geotiff.py default: unw, cc).
IFG_EXTS = ("unw", "cc")
#: Frame metadata LiCSBAS01 fetches. Rasters are clipped; text files are fetched whole.
META_RASTERS = ("E", "N", "U", "hgt")
META_TEXT = ("baselines", "metadata.txt")

#: Buffer added around the twin grid's bounding box (m) so off-fan stable areas and the
#: off-fan GNSS references in the eastern hills fall inside the clip (plan S4 step 5).
CLIP_BUFFER_M = 2000.0

#: Budget (user decision D1 and the S4a task): about 1.4 GB for the primary frame, and the two
#: full-fan frames only if the S4a total stays under about 3 GB.
BUDGET_PRIMARY_BYTES = 1.4e9
BUDGET_TOTAL_BYTES = 3.0e9

RUN_DIR = DATAPREP / "insar_tools" / "s4a_run"          # run config, logs (gitignored)
RUN_CONFIG = RUN_DIR / "run_config.json"
TOOLS_DIR = DATAPREP / "insar_tools"
LICSBAS_DIR = TOOLS_DIR / "LiCSBAS"
LICSBAS_ENV = Path.home() / "miniforge3" / "envs" / "licsbas"
TWIN_GRID = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"

LICENCE = ("Copernicus Sentinel-1 derived-work terms: 'Contains modified Copernicus Sentinel "
           "data [2015-2026] analysed by the COMET LiCSAR system'. Cite LiCSAR: Lazecky M. et "
           "al. (2020) LiCSAR: An Automatic InSAR Tool for Measuring and Monitoring Tectonic "
           "and Volcanic Activity, Remote Sensing 12(15) 2430, doi:10.3390/rs12152430; and, "
           "when processed with LiCSBAS, Morishita Y. et al. (2020) Remote Sensing 12(3) 424, "
           "doi:10.3390/rs12030424. LiCSAR is part of COMET (NERC); hosted on JASMIN.")


def frame_url(frame: str) -> str:
    return f"{LICSAR_PUB}/{int(frame[:3])}/{frame}"

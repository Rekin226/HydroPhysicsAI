"""WP S4b repair round 2 (S4br): settings of the repaired method and of the re-tests.

Written AFTER the diagnosis (``results/dataprep/reports/S4b.repair.md`` Parts 1-2, snapshot
sha256 dbacb416..., 2026-10-17T21:11:20Z). Each change is tied there to a diagnosed defect:

  M1  LiCSBAS12 --multi_prime --nullify, loop threshold pi; LiCSBAS13 --nullify_noloops
  M2  LiCSBAS16 spatio-temporal filter at its defaults (no deramp, no height term, no NaN
      interpolation)
  M3  stated off-fan reference: robust per-epoch offset + height term on the round-1 stable
      strip (no plane)
  M4  GNSS steps: registered catalogue rule (WRA second look); NGL steps.txt equipment
      changes (type 1) in the change span exclude a station (fresh NGL test), as in round 1's
      NGL row; NGL's type-2 entries are distance-based earthquake flags (in 2020-21: M5.5-6.1
      at 86-180 km, static offsets far below 1 mm) and are listed, not applied; S3
      gnss_clean v4 after the fact only (its gate failed)
  M5  second ascending frame 069A_06632_162010 for the full product only (from 2022-11)

The registered G1 rules (``s4b_config``: windows of +-30 d, 20 daily values, 3 epochs, 3 x 3
pixels with >= 4 valid, 7-day median, +-3 d matching, the offset-only leave-one-out tie,
10 mm with >= 5 stations) are imported unchanged. No study-area coordinates live here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402,F401  (registered rules, unchanged)
from s4b_config import DATAPREP, PRIMARY, REPO  # noqa: E402

WP = "S4b"
PREREG_ID = "S4b.insar_repair.v1"
PRODUCT = "insar_069A"
SECOND = "069A_06632_162010"

RUN_DIR = DATAPREP / "insar_tools" / "s4br_run"
EVID = DATAPREP / "reports" / "S4b_evidence" / "repair_2026-10-18"
L0_SECOND = REPO / "data_fetch_api" / "new_measurements" / "licsar_069A_06632" / \
    "fetch_2026-10-18_s4br"

#: Repaired LiCSBAS runs: name -> (frame, first, last yyyymmdd on both epochs of a pair).
RUNS = {
    "P2": (PRIMARY, "20200101", "20221231"),   # pre-origin gate run (U7)
    "F2": (PRIMARY, "20200101", "20251231"),   # full product run, primary frame
    "S2": (SECOND, "20221101", "20251231"),    # full product run, second frame (M5)
}
LOOP_THRE = math.pi                            # M1: = the per-pixel nullification threshold
LICSBAS_STEPS = [
    "LiCSBAS02_ml_prep.py -i GEOC -o GEOCml2 -n 2 --n_para 1",
    "LiCSBAS11_check_unw.py -d GEOCml2 -t TS_GEOCml2",
    f"LiCSBAS12_loop_closure.py -d GEOCml2 -t TS_GEOCml2 -l {LOOP_THRE!r} --multi_prime "
    "--nullify --n_para 1 --nopngs",
    "LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --nullify_noloops --n_para 1 "
    "--mem_size 4000 --nopngs",
    "LiCSBAS14_vel_std.py -t TS_GEOCml2 --mem_size 4000",
    "LiCSBAS15_mask_ts.py -t TS_GEOCml2",
    "LiCSBAS16_filt_ts.py -t TS_GEOCml2 --n_para 1 --nopngs",
]
CUM_FILE = "cum_filt.h5"                       # M2: the product reads the filtered series

#: M3: stated off-fan reference (same stable strip as round 1; plane removed).
REF_TERMS = ("offset", "hgt_km")

#: Fresh NGL re-test (M4 step rule; registered G1 rules otherwise).
NGL_DIRS = [REPO / "data_fetch_api" / "new_measurements" / "ngl_gnss" / d
            for d in ("2026-10-12_s3", "2026-10-12_s3b", "2026-10-12")]
NGL_HOLDINGS = NGL_DIRS[2] / "DataHoldings.txt"
NGL_STEPS = NGL_DIRS[0] / "steps.txt"
#: The only earlier InSAR-vs-NGL comparison (S4b round 1, after the fact) covered
#: 2021-08..2022-12; the fresh window must end before it and start with run P2.
FRESH_FIRST = "2020-01-01"
FRESH_LAST = "2021-07-31"
CHANGE_MONTHS = 14                              # as G1 (2021-09-30 -> 2022-11-30)
NGL_MAX_PIX_DIST_M = 150.0                      # station inside the multilooked grid
NGL_STEP_TYPES = ("1",)                         # equipment changes exclude a station

#: Merging the two frames in the full product (M5): S2 is tied to F2 at their first common
#: epoch, pixel by pixel; where both are valid the product value is their mean.
MERGE_RULE = "mean_where_both_valid_after_tie_at_first_common_epoch"

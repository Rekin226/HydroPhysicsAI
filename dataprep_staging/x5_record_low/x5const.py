"""Constants of WP X5 (route 2 of the 2026-10-25 audit): the 2021 record-low experiment.

Everything a result depends on is fixed here before the declaration is written: windows,
the response-time grids, the support rules, the success rule and the synthetic-check truth.
Changing any value after registration changes the sha256 of this file, which the scorer
checks against the registered declaration and refuses to run on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, REPO, REPORTS_DIR  # noqa: E402

WP = "X5"
PREREG_ID = "X5.record_low_tf.v1"
CODE_DIR = Path(__file__).resolve().parent
OUT = DATAPREP / "x5_record_low"
REPORT_DIR = REPORTS_DIR / PREREG_ID
LOCK = DATAPREP / "preregistration" / ".x5.lock"

# ------------------------------------------------------------------ inputs (read-only)
RING_DDIR = REPO / "chou-shui-data" / "data"          # ls_cache/clean/ls-wra-mlcw-obs__*
RING_SITES = REPO / "results" / "twin" / "mechanism" / "rings_sites.csv"   # 14 sites + nests
HEADS_DIR = DATAPREP / "heads" / "L1_clean" / "v3"
PAGE_COLUMN = (REPO / "results" / "twin_runs" / "stage3_datum_gate" / "coupled_leveling"
               / "vep_zonal_leveling.json")
HEAD_USE = "design_time_evidence"      # guard use for 2012-2021 heads (window_end 2021-12)
HEAD_USE_EXPLORE = "after_the_fact_analysis"   # 2022-2025 heads, exploration only (U7)
HEAD_FLAGS = (0, 1)                    # observed or barometric-corrected; C20 months only
META_COLUMNS = ["sid", "WellDepth"]    # station table: non-personal columns, read explicitly

# ------------------------------------------------------------------ time axis and windows
T0, T1 = "2012-01", "2021-12"         # monthly axis of the main analysis
BURN = ("2012-01", "2013-12")          # sets the initial preconsolidation head (running min)
FIT = ("2014-01", "2019-12")           # ring months used to fit (rings resume 2014-03)
TEST = ("2020-01", "2021-12")          # predicted ring months (record-low years)
ANCHOR = ("2019-07", "2019-12")        # last ring survey in this window re-zeroes the test
PERSIST = ("2018-01", "2019-12")       # persistence baseline: OLS rate of these months
EXPLORE_RATE = (("2020-01", "2021-12"), ("2023-01", "2025-03"))   # U7 exploration only
DT_DAYS = 30.0                         # the twin column's step (compaction.VEPColumn)

# ------------------------------------------------------------------ support rules
MAX_GAP = 6            # head gaps up to this many months are linearly interpolated
MIN_BURN_MONTHS = 12   # observed head months in BURN
MIN_FIT_OBS = 24       # ring months in FIT
MIN_TEST_OBS = 12      # ring months in TEST
MIN_PERSIST_OBS = 12   # ring months in PERSIST
N_SITES = 14

# ------------------------------------------------------------------ estimator
TAU_E_GRID = (0.0, 30.0, 61.0, 91.0, 182.0)                     # days; 0 = instantaneous
TAU_V_GRID = tuple(float(x) for x in np.round(np.geomspace(30.0, 7305.0, 21), 1))  # days

# ------------------------------------------------------------------ success rule (design time)
IMPROVE = 0.80         # TF RMSE <= 0.80 x each baseline's RMSE at a site
MIN_PASS_SITES = 10    # of N_SITES

# ------------------------------------------------------------------ synthetic recovery check
SYN_CASES = (
    {"name": "S1_fast", "ske": 5e-4, "skv": 5e-3, "tau_days": 180.0, "trend_m_yr": 0.003},
    {"name": "S2_slow", "ske": 5e-4, "skv": 1e-2, "tau_days": 730.0, "trend_m_yr": 0.003},
)
SYN_NOISE_M = 0.002          # iid Gaussian noise on each observed ring month (gating)
SYN_NOISE_SENS_M = 0.004     # sensitivity run, reported, not gating
SYN_SEEDS = 20
SYN_SEED0 = 20261025
SYN_D_ID_M = 0.5             # a band is "identifiable" if heads reach >= this below the
                             # burn-in minimum within FIT
SYN_SKV_BAND = (0.67, 1.5)   # median Skv_hat / Skv over seeds
SYN_TAU_BAND = (0.5, 2.0)    # median tau_hat / tau over seeds
SYN_FRAC = 0.80              # of identifiable bands, in each case
SYN_MIN_BANDS = 5            # fewer identifiable bands -> the check is not evaluable (fail)

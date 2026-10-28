"""WP X5 (route 6, Q6 subsidence truth after 2022): fixed paths and the declared rules.

Every rule that the registered test, the synthetic check or the exploratory product uses lives
here, so that the registrations hash one file. No study-area coordinate or data value is
written in this module.

Rules (declared before any leveling value of the 2020-05 / 2021-05 rows is read)
-----------------------------------------------------------------------------
* InSAR: the stated method of insar_069A/v3 (LiCSBAS M1/M2, off-fan offset + height
  reference, LOS / U). The gating series is run P2 (2020-01..2022-12 pairs only, U7); the
  full product (F2 + second frame, ``vertical_ml.npz``) is a declared alternative for the
  test and the base of the exploratory product.
* Valid pixel: LiCSBAS15 mask, finite, and no LiCSBAS13 gap interval since the first epoch
  (flag 0 chain). Point value: median of the 3 x 3 multilooked pixels around the nearest
  pixel (<= 150 m away), >= 4 valid (S4b's station rule).
* GNSS anchors: gnss_clean/full/v4 daily ``value`` (step catalogue removed), flags 0-1, WRA
  stations with role 'member' and NGL stations with role 'fan box' or 'on grid, east of the
  plan box'; 7-day centred median, matched to an InSAR epoch within +-3 days.
* Per-epoch anchoring (primary, 'offset'): D_s(t) = G_s(t) - I_s(t) = k_s + c(t) + e, solved
  by median polish (station constants k_s, epoch offsets c(t); 20 sweeps), epochs with < 5
  stations dropped; anchored field A(x, t) = I(x, t) + c(t) - c(t_ref). Declared alternative
  'plane': after the median polish, c(t) + a(t) E + b(t) N by the S4b two-pass robust least
  squares over the stations of each epoch (>= 6 stations).
* May value: mean of the anchored point values over the epochs inside
  [May 15 - 30 d, May 15 + 30 d] (>= 1 valid epoch); its date = the mean epoch date.
  Primary increment = (A_2021 - A_2020) * 365 / (date_2021 - date_2020 in days) (scaled to
  the nominal leveling interval 2020-05-15 -> 2021-05-15); declared alternative: unscaled.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for p in (str(STAGING), str(STAGING / "s4_insar")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.paths import DATAPREP, PREREG_DIR, REPO, REPORTS_DIR  # noqa: E402,F401

WP = "X5"
CODE_DIR = HERE
OUT = DATAPREP / "x5_insar_anchor"

# ----------------------------------------------------------------------------- inputs
LEV_PANEL = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
LEV_META = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_meta.parquet"
INSAR_V3 = DATAPREP / "insar_069A/v3"
INSAR_ML = INSAR_V3 / "vertical_ml.npz"
P2_DIR = DATAPREP / "insar_tools/s4br_run/P2/TS_GEOCml2"
GNSS = DATAPREP / "gnss_clean/full/v4"
GNSS_STATIONS = GNSS / "stations.csv"
GNSS_WRA = GNSS / "wra/daily.parquet"
GNSS_NGL = GNSS / "ngl/daily.parquet"
PILES = DATAPREP / "deep_benchmarks_printed/v2/printed_monthly.parquet"
COUNTY_MAP = DATAPREP / "zone_county_map/v2/zone_county_map.npz"
PUBLIC_POST = DATAPREP / "subsidence_public_reports/postorigin/v1"

# ----------------------------------------------------------------------------- rules
GNSS_FLAGS = (0, 1)
WRA_ROLES = ("member",)
NGL_ROLES = ("fan box", "on grid, east of the plan box")
GNSS_SMOOTH_DAYS = 7
GNSS_MATCH_TOL_D = 3
MIN_STATIONS_EPOCH = 5
MIN_STATIONS_PLANE = 6
POLISH_SWEEPS = 20
PIX_HALF = 1
PIX_MIN_VALID = 4
PIX_MAX_DIST_M = 150.0
REF_EPOCH = "2020-01-12"

MAY_CENTRE = "05-15"
MAY_HALF_D = 30
NOMINAL_DAYS = 365.0

# ----------------------------------------------------------------------------- test T1
T1_YEARS = (2020, 2021)
LEV_MONTHS = ("2020-05", "2021-05")
LEV_MAX_ABS_MM = 200.0               # gross-error screen on the leveling increment
LEV_BAD_REMARK = ("不列入比較", "異動", "移動")   # monument disturbed (metadata screen)
T1_MIN_N = 200
T1_SLOPE = (0.8, 1.2)
T1_RMSE_MM = 10.0
BLOCK_KM = 5.0                        # spatial blocks of the reported bootstrap
N_BOOT = 2000
SEED_BOOT = 20261025

# ----------------------------------------------------------------------------- synthetic S0
SYN_SEEDS = tuple(range(101, 121))    # 20 seeds
SYN_CM_ABS_MM = (22.0, 32.0)          # quarterly common-mode magnitude, random sign
SYN_CM_JITTER_MM = 5.0                # per-epoch common-mode jitter
SYN_PIX_NOISE_MM = 4.0
SYN_APS_MM = 3.0                      # smooth (about 10 km) per-epoch field
SYN_DRIFT_MM_SQRT_YR = 3.0            # station-site random walk (GNSS vs InSAR)
SYN_N_BAD_STATIONS = 2                # stations drifting +-40 mm/yr
SYN_BAD_RATE_MM_YR = 40.0
SYN_GNSS_NOISE_MM = 5.0
SYN_LEV_NOISE_MM = 3.0
SYN_LEV_DATE_JITTER_D = 20
SYN_SEASONAL_MM = 8.0
#: C1 recovery: median over seeds of slope_ols / lambda in the band, lambda = var(true leveling
#: increment) / var(synthetic leveling increment) on the scored benchmarks (the known
#: errors-in-variables attenuation of an OLS slope on a noisy regressor); C2 median RMSE
#: against the synthetic leveling; C3 the T1 rule (slope 0.8-1.2, RMSE <= 10 mm) in >= 19 of
#: 20 seeds. Raw (unanchored) metrics and the bias are reported, not part of the criterion.
SYN_PASS = {"median_slope_over_lambda": (0.95, 1.05), "median_rmse_mm": 5.0,
            "min_seeds_passing_t1": 19}

# ----------------------------------------------------------------------------- exploration E1
E1_INTERVALS = ((2022, 2023), (2023, 2024), (2024, 2025))
E1_GNSS_LAST = "2025-03-31"
#: deepest sensor first (TKJS(3) 400 m from 2020-08, else TKJS(1) 300 m; NTUH(1) 300 m;
#: SJES 300 m; STES 70 m); the site code is the co-located WRA GNSS station. Printed monthly
#: values are compression (mm, + = compression) per data month, flag 0 only.
PILE_SITES = {"TKJS": ("TKJS(3)", "TKJS(1)"), "STES": ("STES",), "SJES": ("SJES",),
              "NTUH": ("NTUH(1)",)}
PILE_RATE_WINDOW = ("2020-06", "2024-12")
PILE_Q_LAST_MONTHS = 3                # Q(t_L) = mean of the last 3 GNSS months
E1_MIN_STATIONS_PILE = 3              # epochs after the GNSS end (pile anchors only)
CELL_MIN_PIXELS = 5
CELL_AREA_KM2 = 1.0
E1_THRESH_CM = (2.0, 3.0)

PREREG_SYN = "X5.synthetic_anchor.v1"
PREREG_T1 = "X5.insar_leveling_2020_2021.v1"
PREREG_E1 = "X5.post2022_exploration.v1"
T1_TEST_ID = "X5.T1_insar_alone_vs_leveling_may2020_may2021"
T1_UNIT = "x5.lev.may2020_may2021"

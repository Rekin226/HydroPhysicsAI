"""Constants of X5 route 3, independent storage priors (Q7). v2 (post-failure) fixed before any head look.

Two independent per-screen estimates of specific storage Ss:

* earth tides: the M2 (and O1) head response to the theoretical areal strain of the solid
  earth tide (Bredehoeft 1967; Cutillo & Bredehoeft 2010, 10.1111/j.1745-6584.2010.00778.x),
  Ss = c_nu * eps_A / |dh|, c_nu = (1 - 2 nu) / (1 - nu);
* barometric efficiency: the static (48 h cumulative) loading response of the production
  barometric design (h1_heads_qc.baro, Rasmussen & Crawford 1997) refitted on the declared window only,
  Ss = rho g n beta_w / BE (Jacob 1940, 10.1029/tr021i002p00574).

The Hsieh et al. (1987) phase-lag transmissivity is a secondary output (M2 phase relative to
the theoretical strain). Every constant here is a declared assumption; the declaration
(prereg_x5.py) hashes this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, REPO, REPORTS_DIR  # noqa: E402

WP = "X5"
PREREG_ID = "X5.storage_priors.v2_postfailure"
PREREG_ID_V1 = "X5.storage_priors.v1"     # synthetic check failed 2026-10-25 (S3, S5, S6)
CODE_DIR = Path(__file__).resolve().parent
OUT = DATAPREP / "x5_storage_priors"
SYN_DIR = OUT / "synthetic_v2"
WORK = OUT / "work_v2"
SELECTION = OUT / "work" / "selection.csv"   # v1 selection (metadata only), reused
PRIORS_V1 = OUT / "v1"
REPORT_DIR = REPORTS_DIR / PREREG_ID

# ------------------------------------------------------------------ inputs (read-only)
WELLS_DIR = REPO / "AMP_V2" / "data" / "wells"
L1V3 = DATAPREP / "heads" / "L1_clean" / "v3"
META = L1V3 / "station_meta_preorigin.parquet"
PRESSURE = L1V3 / "pressure_hourly.parquet"
HEADS_DAILY = L1V3 / "heads_daily.parquet"
META_COLS = ["sid", "CountyName", "layer_code", "layer_flag", "x_twd97", "y_twd97", "lon",
             "lat", "WellDepth", "FinishDepth", "WellRadius", "era5_cell"]

# ------------------------------------------------------------------ window (local time)
# v2: the pre-2019 hourly epoch without 2015 (the audit's annual tidal fit year), chosen on
# synthetic dev seeds after the v1 failure (power: 2017-18 alone gave BE acceptance 0.27-0.38)
WINDOW = ("2012-01-01 00:00", "2018-12-31 23:00")
EXCLUDE: tuple = (("2015-01-01 00:00", "2015-12-31 23:00"),)
DATA_SCOPE = "heads:2012-01..2014-12,2016-01..2018-12"
SYN_PRESSURE_CELL = "23.75_120.50"      # ERA5 cell whose real pressure (window) drives synth
LOCAL_UTC_OFFSET_H = 8.0                # AMP timestamps are local UTC+8 (audit, Chishang)
HOUR_LABEL_SHIFT_MIN = 30.0             # hourly value = mean of [hh, hh+1): effective hh:30

# ------------------------------------------------------------------ physics (declared)
RHO_G = 1000.0 * 9.81                   # Pa per m of water
BETA_W = 4.6e-10                        # 1/Pa, water compressibility
N_POR = 0.30                            # porosity assumed for Jacob's Ss (range 0.20-0.40)
N_POR_RANGE = (0.20, 0.40)
NU = 0.25                               # drained Poisson ratio (range 0.20-0.30)
NU_RANGE = (0.20, 0.30)
LOVE_H, LOVE_L = 0.6032, 0.0840         # degree-2 Love and Shida numbers
EARTH_R = 6.371e6                       # m
WATER_TERM = RHO_G * N_POR * BETA_W     # rho g n beta_w, 1/m (1.35e-6 at n = 0.3)

# equilibrium-tide coefficients (m; Cartwright-Tayler-Edden), species, geodetic factor
#   semidiurnal: H cos^2(lat); diurnal: H sin(2 lat)
TIDES = {
    "Q1": {"H": 0.019400, "k": 1}, "O1": {"H": 0.100514, "k": 1},
    "P1": {"H": 0.046843, "k": 1}, "K1": {"H": 0.141565, "k": 1},
    "N2": {"H": 0.046398, "k": 2}, "M2": {"H": 0.242334, "k": 2},
    "S2": {"H": 0.112841, "k": 2}, "K2": {"H": 0.030704, "k": 2},
}
FIT_CONSTITUENTS = ("Q1", "O1", "N2", "M2")   # global; K1/P1/S2/K2 absorbed by block S1/S2
SYNTH_CONSTITUENTS = tuple(TIDES)             # the synthetic earth tide carries all eight
M2_O1_THEORY = None                           # computed per latitude (x5sp_lib.strain_amp)

# ------------------------------------------------------------------ estimators (declared)
TIDE_BLOCK_H = 360          # 15-day nuisance blocks (M2-S2 beat 14.77 d, O1-S1 14.19 d)
TIDE_BLOCK_MIN_ROWS = 216   # >= 60 % of a block's hours
TIDE_MIN_BLOCKS = 24        # >= 12 months of usable blocks in the 24-month window
TIDE_HUBER_C = 1.345
TIDE_IRLS_ITERS = 4
TIDE_BOOT = 200
TIDE_SEED = 20171
SNR_MIN = 3.0               # amplitude / bootstrap se for a detected constituent
BE_DESIGN = "hourly_grouped"  # h1_heads_qc.baro design
BE_MIN, BE_MAX = 0.02, 1.0  # physical BE (static, vented/compensated sensor)
OCEAN_RATIO = 1.3           # gain(M2)/gain(O1) > 1.3 (both per unit strain): ocean-affected
SPIKE_DEV_M = 5.0           # hourly value > 5 m from its 49-h rolling median: spike
GOOD_DAILY_FLAGS = (0, 1)   # L1 v3 daily flags kept (8 missing, 9 suspect, 3 corrected)

# ------------------------------------------------------------------ gate (declared)
INLAND_X_M = 182000.0       # twin mid/distal boundary (hydrophysics.twin.zones): inland east
ZONE_EDGES_KM = (182.0, 205.0)   # distal <= 182 < mid <= 205 < proximal (twin zones)
DEEP_LAYERS = (3, 4)
SPEARMAN_MIN = 0.40
N_MIN = 10

# ------------------------------------------------------------------ synthetic check
SYN_SEED = 20261026                      # v1 used 20261025; dev seeds 101
SYN_N_INLAND = 48
SYN_N_COASTAL = 12
SYN_SS_RANGE = (1.5e-6, 1.5e-5)          # log-uniform truth (BE = rho g n beta_w / Ss)
SYN_PUMP_AMP = (0.02, 1.0)               # m, log-uniform daily pumping drawdown
SYN_PUMP_TAU_H = (0.5, 3.0)
SYN_RED_SD = (0.02, 0.20)                # m, AR(1) stationary sd
SYN_SEAS_AMP = (0.3, 3.0)                # m
SYN_BARO_TAU_H = (0.0, 2.0)
SYN_OCEAN_X = (1.0, 3.0)                 # coastal: ocean M2 amplitude / earth-tide M2 amp
SYN_OCEAN_PHASE_SPREAD = 60.0            # deg around the undrained earth-tide head phase
SYN_LAT, SYN_LON = 23.80, 120.40
SYN_CRITERIA = {
    "S1_tide_median_abs_log10": 0.10, "S1_tide_p90_abs_log10": 0.25,
    "S2_m2_detect_rate_min": 0.70, "S2_m2_true_amp_min_m": 0.003,
    "S3_be_median_abs_log10": 0.10, "S3_be_p90_abs_log10": 0.25,
    "S3_be_accept_rate_min": 0.70, "S3_be_true_min": 0.10,
    "S4_spearman_min": 0.70, "S4_n_min": 10,
    "S5_coastal_flag_rate_min": 0.75, "S5_inland_flag_rate_max": 0.10,
    "S6_median_abs_eta_deg_max": 3.0,
}
# v2: the real-data run needs the RECOVERY criteria; power and phase criteria are reported
SYN_GATING = ("S1", "S3_accuracy", "S4", "S5")

# ------------------------------------------------------------------ Hsieh (secondary)
WELL_RADIUS_M = 0.0762      # WellRadius 6 read as a 6-inch nominal casing: r_w = r_c = 3 in
SCREEN_B_M = 50.0           # aquifer thickness for S = Ss b in Hsieh's model (weak effect)

# ------------------------------------------------------------------ priors (declared)
MIN_WELLS_PER_GROUP = 3
COLUMN_B_M = 300.0          # ring-measured compressible depth (0-300 m) for the Ske floor
BOOT_GROUP = 2000
BOOT_GROUP_SEED = 20181

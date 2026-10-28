"""Fixed constants of WP X5 (declared in X5.changhua_water.v1 before any score)."""
from __future__ import annotations

from common.paths import DATAPREP, L0_NEW, REPO, REPORTS_DIR, STAGING

WP = "X5"
PREREG_ID = "X5.changhua_water.v1"
CHECK_ID = "X5C.yunlin_heldout_agreement"
LOCK = DATAPREP / "preregistration" / ".x5.lock"

CODE_DIR = STAGING / "x5_changhua_water"
OUT = DATAPREP / "x5_changhua_water"
WORK = OUT / "work"
SYN_DIR = OUT / "synthetic"
FIT_DIR = OUT / "fit"
REPORT_DIR = REPORTS_DIR / PREREG_ID

# --- inputs (read-only)
MOA_CSV = REPO / "chou-shui-data/data/water/irrigation/moa_35644_actual.csv"
JIJI_CSV = REPO / "chou-shui-data/data/water/jiji_weir_annual_supply.csv"
W1_PRE = DATAPREP / "surface_water_v4/preorigin/v2"
W1_FULL = DATAPREP / "surface_water_v4/full/v4"
X4_PRE = DATAPREP / "canal_redistribution_x4/preorigin/v1/canal_fields_x4.npz"
V3_NPZ = REPO / "results/twin/surface_water_v3.npz"
COUNTY_MAP = DATAPREP / "zone_county_map/v2/zone_county_map.npz"
HEADS = DATAPREP / "heads/L1_clean/v3"
LICSAR = L0_NEW / "licsar_069A_06694"

# --- design (declared)
YEARS_ALL = list(range(2012, 2026))          # product years (2025 annual extrapolated)
CALIB_YEARS = list(range(2015, 2022))        # Yunlin timing calibration (design-time years)
HELDOUT_YEARS = [2012, 2013, 2014, 2022]     # held-out Yunlin years, all pre-2023 (U7)
EXPLORE_YEARS = [2023, 2024, 2025]           # exploration only (U7)
SD_DEF_YEARS = list(range(2016, 2023))       # Yunlin office-minus-Jiji scatter (pre-origin)
LEVEL_YEARS = list(range(2016, 2023))        # post-2016 reporting basis, pre-origin
SHORT_PAIR_DAYS = 12
RIVER_MONTHS = (1, 2, 3, 4, 5)               # dry season: the weir takes almost the river
HEAD_MIN_WELLS = 5
THRESHOLD_R = 0.6
INDICATORS_T3 = ("coh", "riv", "head")       # primary: three indicators
INDICATORS_T2 = ("coh", "riv")               # declared alternative / rung input (no heads)
COUNTY = {"chc": "彰化縣", "yl": "雲林縣"}
MOA_OFFICE = {"chc": "彰化", "yl": "雲林"}

# --- synthetic recovery check (declared)
SYN_B = {"coh": 0.4, "riv": 0.3, "head": 0.2}
SYN_NOISE_SD = 0.2
SYN_SEED = 20261025
# v3's published crop calendar weights (results/twin/surface_water_v3.npz 'calendar')
SYN_CALENDAR = {1: 0.0, 2: 1.5, 3: 1.5, 4: 1.0, 5: 1.0, 6: 0.5, 7: 1.5, 8: 1.0, 9: 1.0,
                10: 1.0, 11: 0.5, 12: 0.0}
SYN_PASS = {"b_rel_tol": 0.5, "heldout_r_min": 0.9, "anomaly_r_min": 0.5,
            "annual_rel_tol": 1e-9}

SECONDS_PER_DAY = 86400.0
CELL_AREA_M2 = 1.0e6
__all__ = [k for k in dir() if k.isupper()] + ["REPO"]

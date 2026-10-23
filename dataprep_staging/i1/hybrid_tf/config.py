"""Paths, frozen hyperparameters and driver-availability lags of the hybrid TF layer.

The hyperparameters are those of ``dataprep_staging/i1/diag_changhua_drivers.py`` (filters of
0, 2, 6 and 18 months, ridge alpha 1). They are copied, not tuned: no number in this package
chooses them (U7). The availability lags come from :mod:`.inventory` (task 1) for a test window
that starts in 2026-11 with heads fetched to 2026-09.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REL = REPO / "results/dataprep/model_ready/release_20261012b2"
OUT = REPO / "results/dataprep/reports/I1_hybrid_tf"
DRAFT = OUT / "draft_declaration"

# R0 (design-time folds, batch-8 recipe with the coordinate fix): out-of-fold heads per well.
R0_RUN = REPO / "results/twin_runs/stage3_datum_gate_cwa"
# Products with a manifest (read through common.guard.open_product).
HEADS_PRE = REL / "preorigin/heads_observed/v1"
ELEC_PRE = REL / "preorigin/electricity_census/v1"
SW_PRE = REL / "preorigin/surface_water/v1"
PW_PRE = REL / "preorigin/public_wells_member/v1"
ET0_PRE = REL / "preorigin/et0/v1"
W1_PRE = REPO / "results/dataprep/surface_water_v4/preorigin/v2"   # river gauges (L1 daily)
COUNTY_MAP = REPO / "results/dataprep/zone_county_map/v2"
# Inputs without a manifest (the twin's rain cache, rebuilt from CODIS; read directly).
RAIN = REPO / "chou-shui-data/data/rf_timeseries.csv"
RAIN_STATIONS = REPO / "chou-shui-data/data/rf_stations.csv"
GW_STATIONS = REPO / "chou-shui-data/data/gw_stations.csv"   # locations of the 61 ET0 series

COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
GAUGES = {"river_apex_H057": "H057", "river_midfan_H071": "H071"}
DRIVERS = ("electricity_local_3km", "electricity_county", "rain_recharge", "canal_w1_cell",
           "canal_district", "river_apex_H057", "river_midfan_H071", "public_wells_county")

# Frozen hyperparameters (copied from the I1_changhua_drivers design; never tuned here).
TAUS = (0, 2, 6, 18)          # causal exponential response times, months
ALPHA = 1.0                   # ridge penalty on standardised features
LOCAL_RADIUS_M = 3000.0       # radius of the local electricity sum
MIN_TRAIN = 24                # minimum observed training months per well and fold
MIN_SCORED = 24               # minimum scored months per well (pooled over folds)
MAX_ORIGIN_GAP = 3            # an origin observation may be at most 3 months before the origin
RHO_MIN_PAIRS = 24            # lag-1 coefficient needs >= 24 consecutive observed pairs

# Design window: heads 2012-02..2022-12 (R0's scored months), drivers from 2012-01.
DRIVER_START, DESIGN_END = "2012-01", "2022-12"
HEAD_START = "2012-02"

# Forecast setting of the draft test (task 3): window 2026-11..2027-10, heads fetched to
# 2026-09 (origin), so the head origin is 2 months before the first forecast month.
ORIGIN_LAG = 2
# Months before the first forecast month from which each driver is not yet recorded when the
# predictions are frozen (inventory, availability.json; last recorded month -> lag):
# electricity 2026-06 (2026-07 partial) -> 4; rain and ET0 2026-09 -> 1; canal 2025-12 (the
# 2026 months are a flag-6 rain-driven extrapolation, not a record) -> 10; river yearbook
# 2025-12 -> 10; public wells member 2022-12 -> 46 (held at its last value).
# Correction 2026-10-20: the canal lag was first set to 1 before the inventory ran; design
# run 4 uses 10.
AVAIL_LAG = {"electricity_local_3km": 4, "electricity_county": 4, "rain_recharge": 1,
             "canal_w1_cell": 10, "canal_district": 10, "river_apex_H057": 10,
             "river_midfan_H071": 10, "public_wells_county": 46}
# Fill rule after a driver's last recorded month, fixed before any score was read:
# month-of-year climatology of the fit window, except the public wells (a step schedule with
# no seasonal cycle), which hold their last value.
HOLD_LAST = frozenset({"public_wells_county"})
# Drivers that a one-month-ahead scorer could have recorded during the window (CODIS rain and
# Open-Meteo ET0 are re-fetchable each month); the others keep the seasonal cut-offs.
RECORDED_IN_WINDOW = frozenset({"rain_recharge"})
# A rain month is recorded when it has ended and >= 5 of the 26 gauges have >= 90 % of days.
RAIN_MIN_GAUGES = 5
# A river month is recorded when the gauge has >= 20 daily values in it.
RIVER_MIN_DAYS = 20
# A gridded driver month is recorded when >= 99 % of its (active) cells carry flags 0-5.
GRID_MIN_SHARE = 0.99

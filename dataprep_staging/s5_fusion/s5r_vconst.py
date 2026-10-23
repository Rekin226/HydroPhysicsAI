"""WP S5 repair round: constants of the subsidence_fused/v2 product rules (declaration
``S5.subsidence_fused_rules.v2``) and of the after-the-fact repair diagnostics."""
from __future__ import annotations

# ----------------------------------------------------------------------------- GNSS (amendment 1)
#: a gap longer than this between consecutive usable GNSS days starts a new offset segment
#: (gnss_clean/v2: S3's ``Params.tie_max_gap``; S3's step scan cannot test a step across it)
GNSS_GAP_DAYS = 30
GNSS_USABLE_DAY_FLAGS = (0, 1, 3)              # days that count as data (S1b's gap definition)
GNSS_DAILY = {"wra": "wra/daily.parquet", "ngl": "ngl/daily.parquet",
              "ncku": "wra/ncku_daily.parquet"}

# ----------------------------------------------------------------------------- leveling (am. 2)
STEP_Z = 5.0                                   # |step| / sd above this ...
STEP_MIN_MM = 30.0                             # ... and |step| at least this flags a benchmark
STEP_MIN_SIDE = 2                              # surveys needed on each side of a step
STEP_MIN_SURVEYS = 4                           # benchmarks with fewer surveys are not scanned

# ----------------------------------------------------------------------------- diagnostics
DIAG_STAGE_A_MAXFEV = 300                      # continuation of the stage-A optimiser
T2B_BUFFER_KM = 10.0                           # after-the-fact T2 with a buffer (plan 4.3)
T2B_IDW_RADIUS_KM = 20.0                       # IDW neighbours beyond the buffer, within this
CAMPAIGN_SD_START = 3.0
SJES = ("wra", "SJES")

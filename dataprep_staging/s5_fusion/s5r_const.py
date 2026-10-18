"""WP S5 repair round (2026-10-17): constants of the stage-2 monthly anomaly and of the
registered monthly-field test ``S5.monthly_field.v1`` (hashed by that registration; frozen).

The v2 product rules (GNSS offset segments, stepped benchmarks, diagnostics) have their own
constants in ``s5r_vconst.py``. The round-1 modules (s5const, s5model, s5data, s5fit, s5hyper,
s5tests, s5insarP, s5looks, s5prereg and tests/test_s5model.py) are hashed by
``S5.subsidence_fusion.v1`` and by subsidence_fused/v1 and stay unchanged; the repair code
imports them read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import s5const as C  # noqa: E402

DECL_ID = "S5.subsidence_fused_rules.v2"       # declaration of the v2 product rules
MTEST_ID = "S5.monthly_field.v1"               # registered test of the monthly component
MTEST_TEST = "S5.M1_monthly_anomaly_vs_wra_gnss"
EVIDENCE_R = C.EVIDENCE / "repair"

# ----------------------------------------------------------------------------- stage 2
S2_NAMES = ("sig_ins2", "sig_aps2", "ell_aps2", "sig_a", "ell_a", "rho")
S2_BOUNDS = {"sig_ins2": (0.2, 40.0), "sig_aps2": (0.05, 40.0), "ell_aps2": (2.0, 40.0),
             "sig_a": (0.05, 40.0), "ell_a": (4.0, 60.0), "rho": (0.0, 0.99)}
S2_NOISE = ("sig_ins2", "sig_aps2", "ell_aps2")
S2_MAXFEV_NULL = 600                           # noise-only model first ...
S2_MAXFEV = 800                                # ... then the full model from its optimum ...
S2_MAXFEV_RESTART = 400                        # ... and one Nelder-Mead restart
S2_START = {"sig_a": 3.0, "rho": 0.5}          # ell_a starts at the annual ell_km
S2_JITTER = 1e-7
S2_MIN_FOOTPRINT = 30                          # cells needed to define a month's plane
VR_A_MIN = 0.5                                 # support: plane-free anomaly variance cut >= 50 %
MONTHLY_FIRST = "2020-01"                      # InSAR era of the product (run F)
MONTHLY_LAST = "2025-12"

# ----------------------------------------------------------------------------- test M1
M1_WINDOW = ("2020-01", "2022-12")
M1_FAMILY = "wra"
M1_PLANE_MIN_OTHERS = 6                        # months need >= 7 stations
M1_MIN_MONTHS = 12                             # station months in the window to enter
M1_MIN_TARGETS = 100                           # evaluable: scored station-months ...
M1_MIN_STATIONS = 8                            # ... at this many stations
M1_SEED = 5006
M1_BOOT_P_MAX = 0.10                           # skill: bootstrap P(ratio >= 1) at most this
Z90 = 1.6448536269514722

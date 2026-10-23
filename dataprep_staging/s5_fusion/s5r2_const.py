"""WP S5 repair round 2 (S5r): constants of the repaired model, its hyperparameter procedure H2
and the re-tests. Written after the diagnosis (reports/S5.repair.md Part 1, snapshot sha256
9340a558...). Every change is tied there (Part 2) to a diagnosed defect:

  R1 (D1)  temporal prior: long-term velocity + white knot-velocity anomalies + reverting
           displacement anomalies at the knots, replacing the integrated random walk
  R2 (D4)  leveling campaign offsets (county x survey month), prior N(0, sig_camp^2)
  R3 (D2)  GNSS: station-level exponentially correlated noise (sig_gc, tau_g) in addition to the
           white monthly noise
  R4 (D2)  rings and piles: sensor-level exponentially correlated noise (sig_cc, tau_c)
  R5 (D2)  InSAR: per-unit (4 km block x segment) random rate, prior N(0, sig_ir^2)
  R6 (D3)  inputs: gnss_clean/preorigin/v4 and full/v4 (with the 30-day gap-segment rule on the
           v4 daily records), InSAR run P2 (pre-origin) and insar_069A/v3 (product)

Thresholds, VR fraction, coverage rule, baselines and windows of the registered tests are
imported from s5const unchanged.
"""
from __future__ import annotations

import s5const as C

WP = C.WP
EVID = C.EVIDENCE / "repair_r2"
WORK = EVID / "work"
PREREG_ID = "S5.repair_r2.v1"

GNSS_PRE_V4 = C.DATAPREP / "gnss_clean" / "preorigin" / "v4"
GNSS_FULL_V4 = C.DATAPREP / "gnss_clean" / "full" / "v4"
INSAR_V3 = C.DATAPREP / "insar_069A" / "v3"
INSAR_P2_CELLS = WORK / "insar_runP2_cells.npz"

# ----------------------------------------------------------------------------- hyperparameters
HYPER_START = {
    "ell_km": 12.0, "sig_v0": 15.0, "sig_be": 5.0, "sig_ep": 5.0, "sig_s": 5.0,
    "sig_lev": 4.0, "sig_rb": 3.0, "sig_camp": 2.0,
    "sig_gnss": 3.0, "sig_d": 3.0, "sig_cm": 2.0, "sig_gc": 4.0, "tau_g": 0.5,
    "sig_ins": 4.0, "sig_aps": 4.0, "ell_aps_km": 8.0, "sig_ir": 2.0,
    "sig_cmp": 2.0, "sig_rc": 2.0, "sig_cc": 4.0, "tau_c": 0.5,
}
HYPER_BOUNDS = {
    "ell_km": (4.0, 60.0), "sig_v0": (1.0, 100.0), "sig_be": (0.1, 60.0),
    "sig_ep": (0.1, 40.0), "sig_s": (0.1, 30.0), "sig_lev": (0.5, 40.0),
    "sig_rb": (0.05, 30.0), "sig_camp": (0.05, 20.0),
    "sig_gnss": (0.5, 30.0), "sig_d": (0.05, 30.0), "sig_cm": (0.05, 20.0),
    "sig_gc": (0.05, 40.0), "tau_g": (0.05, 10.0),
    "sig_ins": (0.2, 40.0), "sig_aps": (0.05, 40.0), "ell_aps_km": (2.0, 40.0),
    "sig_ir": (0.05, 30.0),
    "sig_cmp": (0.2, 30.0), "sig_rc": (0.05, 30.0), "sig_cc": (0.05, 40.0),
    "tau_c": (0.05, 10.0),
}
STAGES = {
    "A": ("ell_km", "sig_v0", "sig_be", "sig_ep", "sig_s", "sig_lev", "sig_rb", "sig_camp"),
    "B": ("sig_gnss", "sig_d", "sig_cm", "sig_gc", "tau_g"),
    "C": ("sig_ins", "sig_aps", "ell_aps_km", "sig_ir"),
    "D": ("sig_cmp", "sig_rc", "sig_cc", "tau_c"),
}
#: Nelder-Mead budget per stage: first pass, then one restart from the best point
NM_MAXFEV = {"A": 400, "B": 200, "C": 150, "D": 150}
NM_RESTART = {"A": 150, "B": 80, "C": 80, "D": 80}

# ----------------------------------------------------------------------------- fresh test T1F
#: pseudo-origin 2020-01: leveling 2018-01..2019-12 withheld (never an S5 temporal target);
#: training leveling <= 2017-12; GNSS, rings, piles <= 2019-12; no InSAR (none before 2020)
T1F_WITHHELD = ("2018-01-01", "2019-12-31")
T1F_TRAIN_END = "2017-12-31"
T1F_OTHER_END = "2019-12-31"
T1F_BASE_WIN = ("2012-01-01", "2017-12-31")
KNOTS_T1F_A = tuple(float(y) for y in range(2012, 2019))     # stage A, leveling <= 2017
KNOTS_T1F = tuple(float(y) for y in range(2012, 2021))       # fits with data <= 2019-12
KNOTS_A = tuple(float(y) for y in range(2012, 2023))         # main stage A (as round 1)

# ----------------------------------------------------------------------------- T2 with buffer
T2B_BUFFER_KM = 10.0       # plan 4.3: >= 10 km buffer for leveling
T2B_IDW_RADIUS_KM = 20.0   # IDW neighbours beyond the buffer, within this (repair round 1)

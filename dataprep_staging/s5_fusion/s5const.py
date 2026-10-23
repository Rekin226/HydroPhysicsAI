"""WP S5 constants: paths, model structure, windows, QC rules, tests and thresholds.

Everything a registered test depends on is fixed here, before any fit to real data, and this
file is hashed by the pre-registration ``S5.subsidence_fusion.v1``.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, REPO, TWIN_GRID_NPZ  # noqa: E402

WP = "S5"
PRODUCT = "subsidence_fused"
PREREG_ID = "S5.subsidence_fusion.v1"
EVIDENCE = DATAPREP / "reports" / "S5_evidence"
WORK = EVIDENCE / "work"

# ----------------------------------------------------------------------------- inputs (L0, products)
LEVELING_PANEL = REPO / "chou-shui-data" / "data" / "ls_cache" / "ls-wra-lsp-obs__choushui_panel.parquet"
RING_STATIONS = REPO / "chou-shui-data" / "data" / "mlcw_stations.csv"
RING_DIR = REPO / "data_fetch_api" / "new_measurements" / "compaction"
GNSS_PRE = DATAPREP / "gnss_clean" / "preorigin" / "v2"
GNSS_FULL = DATAPREP / "gnss_clean" / "full" / "v2"
INSAR_V2 = DATAPREP / "insar_069A" / "v2"
INSAR_RUN_DIR = DATAPREP / "insar_tools" / "s4b_run"
INSAR_GAPS_P = INSAR_RUN_DIR / "gaps_P.npz"
INSAR_CUM_P = INSAR_RUN_DIR / "P" / "TS_GEOCml2" / "cum.h5"
PILES = DATAPREP / "deep_benchmarks_printed" / "v2"
RINGS_BRIDGED = DATAPREP / "rings_bridged" / "v2"        # interval definition (top/bottom ring)
TWIN_GRID = TWIN_GRID_NPZ
UNITS_V3 = DATAPREP / "validation_units" / "v3" / "units.json"

# ----------------------------------------------------------------------------- time
T_REF = 2012.0                     # u(x, T_REF) = 0 by definition (displacement reference)
KNOTS_FULL = tuple(float(y) for y in range(2012, 2028))       # Jan 1 2012 .. Jan 1 2027
KNOTS_PRE = tuple(float(y) for y in range(2012, 2024))        # Jan 1 2012 .. Jan 1 2023
ORIGIN_END = "2022-12-31"          # last pre-origin day
LEVELING_START = "2012-01-01"
LEVELING_END = "2021-12-31"
PRODUCT_YEARS = tuple(range(2012, 2027))                      # annual increments 2012..2026
EXTRAPOLATED_FROM = 2026           # year 2026 has almost no data: flag 6

# ----------------------------------------------------------------------------- space
LATTICE_KM = 4.0                   # spacing of the latent node lattice (km)
LATTICE_REACH_KM = 6.0             # keep nodes within this distance of an active cell centre
MATERN_NU = 1.5                    # smoothness (fixed)
SITE_REACH_KM = 3.0                # a point instrument is used if within this of an active cell
INSAR_BLOCK = 4                    # InSAR aggregated over 4 x 4 twin cells (4 km blocks)
INSAR_BLOCK_MIN_CELLS = 4          # cells with data needed in a block-segment
INSAR_CELL_COVER = 0.9             # a cell enters a block-segment if valid in >= 90 % of epochs
INSAR_EPOCH_COVER = 0.8            # ... and a block-epoch needs >= 80 % of those cells

# ----------------------------------------------------------------------------- QC
LEV_MIN_SURVEYS = 2                # surveys in the window for a benchmark to enter
LEV_MAX_RATE_M_YR = 0.5            # survey-to-survey |rate| above this drops the benchmark (reset)
OUTLIER_Z = 5.0                    # one re-fit after dropping |standardised residual| > 5
RING_SEGMENT_GAP_DAYS = 180        # a survey gap longer than this starts a new ring segment
RING_COVER = 0.8                   # bottom ring: deepest ring read in >= 80 % of 2019-2021 surveys
RING_PRE_WIN = ("2019-01-01", "2021-12-31")

# ----------------------------------------------------------------------------- hyperparameters
#: starting values (log-space optimisation, Nelder-Mead) and bounds; units mm, yr, km
HYPER_START = {
    "ell_km": 12.0, "sig_v0": 15.0, "q": 4.0, "sig_s": 5.0, "sig_lev": 6.0, "sig_rb": 3.0,
    "sig_gnss": 4.0, "sig_d": 3.0, "sig_cm": 2.0,
    "sig_ins": 4.0, "sig_aps": 4.0, "ell_aps_km": 8.0,
    "sig_cmp": 2.0, "sig_rc": 2.0,
}
HYPER_BOUNDS = {
    "ell_km": (4.0, 60.0), "sig_v0": (1.0, 100.0), "q": (0.01, 400.0), "sig_s": (0.1, 30.0),
    "sig_lev": (0.5, 40.0), "sig_rb": (0.05, 30.0),
    "sig_gnss": (0.5, 30.0), "sig_d": (0.05, 30.0), "sig_cm": (0.05, 20.0),
    "sig_ins": (0.2, 40.0), "sig_aps": (0.05, 40.0), "ell_aps_km": (2.0, 40.0),
    "sig_cmp": (0.2, 30.0), "sig_rc": (0.05, 30.0),
}
STAGES = {
    "A": ("ell_km", "sig_v0", "q", "sig_s", "sig_lev", "sig_rb"),       # leveling only
    "B": ("sig_gnss", "sig_d", "sig_cm"),                                # + GNSS
    "C": ("sig_ins", "sig_aps", "ell_aps_km"),                           # + InSAR (run P)
    "D": ("sig_cmp", "sig_rc"),                                          # + rings, piles
}
NM_MAXFEV = {"A": 220, "B": 120, "C": 120, "D": 80}
VAGUE_SD = 1.0e4                   # sd of diffuse nuisance parameters (mm or mm/yr)
MU_V_SD = 100.0                    # prior sd of the fan-wide mean velocity (mm/yr)
FRAME_RATE_SD = 50.0               # prior sd of a GNSS family frame rate (mm/yr)
F_BOUNDS = (0.2, 1.2)              # partial-column factor of a ring or pile

# ----------------------------------------------------------------------------- tests
VR_MIN = 0.5                       # plan: scored only where post-gap data cut variance >= 50 %
COVERAGE_LEVEL = 0.90
T1_WITHHELD = ("2020-01-01", "2021-12-31")     # leveling withheld (mirror of the real gap)
T1_BASE_WIN = ("2012-01-01", "2019-12-31")     # baseline rate window
T1_BASE_MIN_SURVEYS = 3
T1_BASE_MIN_SPAN_YR = 3.0
T1_MIN_SCORED = 20                 # masked benchmarks needed for T1 to be evaluable
T2_CLUSTER_KM = 10.0               # complete-linkage diameter of the leveling blocks
T2_IDW_RADIUS_KM = 10.0
T2_IDW_POWER = 2.0
T2_IDW_MIN_NEIGHBOURS = 3
SEED = 5005
BOOT_N = 2000

#: after-the-fact (U7: report only) families withheld after 2022-12
AFTER_FAMILIES = ("gnss_wra", "gnss_ngl", "insar", "piles", "rings")

"""Constants of WP N1: sources, event windows, reference years, paths (fixed before any test).

Event windows are the plan's (WP N1 'Gap'): the 2021 drought delivery cut and the 2015 and
2020 spring shortfalls, each scored on January-May of the event year (the W1 benchmark's
dry-season window). Reference years for the per-well seasonal climatology and trend are the
pre-origin years that are neither an event year nor the 2012 start-up year (12-month burn-in,
plan P4 fact 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, L0_NEW, REPO, REPORTS_DIR  # noqa: E402

WP = "N1"
PRODUCT = "natural_experiments"
CODE_DIR = Path(__file__).resolve().parent
EVID = REPORTS_DIR / "N1_evidence"
API_CACHE = EVID / "api_cache"

# ------------------------------------------------------------------ L0 sources (read-only)
L0_MISC = L0_NEW / "wra_reports_misc"
SRC = {
    "moea_annex": L0_MISC / "moea2019_yunchang_plan_annex.pdf",
    "cy53001": L0_MISC / "cy53001.pdf",
    "drought_wells": L0_MISC / "wra111_drought_wells_report.pdf",
    "brief108": L0_MISC / "wra_cy_subsidence_results_108.pdf",
    "brief109": L0_MISC / "wra_cy_subsidence_results_109.pdf",
    "brief110": L0_MISC / "wra_cy_subsidence_results_110.pdf",
    "brief111": L0_MISC / "wra_cy_subsidence_results_111.pdf",
    "brief112": L0_MISC / "wra_cy_subsidence_results_112.pdf",
}

# Chart pages transcribed with the Claude API (decision D6 b; N1 approved for transcription).
CHART_PAGES = [
    {"key": "b110_p10", "src": "brief110", "page": 10, "county": "Changhua",
     "years": [2011, 2021]},
    {"key": "b110_p19", "src": "brief110", "page": 19, "county": "Yunlin",
     "years": [2011, 2021]},
    {"key": "b112_p23", "src": "brief112", "page": 23, "county": "Changhua",
     "years": [2011, 2023]},
    {"key": "b112_p26", "src": "brief112", "page": 26, "county": "Yunlin",
     "years": [2011, 2023]},
]
MODEL_PDF = "claude-opus-5-5"      # first read: the page as a PDF document
MODEL_IMG = "claude-sonnet-5-5"    # second, independent read: the page rendered as PNG
RENDER_DPI = 200

# ------------------------------------------------------------------ products used
TWIN_THETA = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
TWIN_FOLD_THETAS = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_fold_thetas.json"
TWIN_KFOLD_WELLS = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_kfold_wells.csv"
ENGINE_DIR = DATAPREP / "changhua_abstraction/engine/v2"
HEADS_DIR = DATAPREP / "heads/L1_clean/v3"
W1_PRE = DATAPREP / "surface_water_v4/preorigin/v2"
C0_PRE = DATAPREP / "pump_electricity/preorigin/v3"
PILES_PRINTED = DATAPREP / "deep_benchmarks_printed/v2"
UNITS_V3 = DATAPREP / "validation_units/v3"

# ------------------------------------------------------------------ test design (fixed)
EVENTS = {
    "E2015": {"year": 2015, "months": [1, 2, 3, 4, 5],
              "what": "2015 spring delivery shortfall (Jiji Jan-May supply 1.54 / 1.26 x 10^8 m3 "
                      "Changhua / Yunlin irrigation)"},
    "E2020": {"year": 2020, "months": [1, 2, 3, 4, 5],
              "what": "2020 spring delivery shortfall (2.03 / 1.26 x 10^8 m3)"},
    "E2021": {"year": 2021, "months": [1, 2, 3, 4, 5],
              "what": "2021 drought crop-1 rotation / delivery cut (1.43 / 0.79 x 10^8 m3)"},
}
REFERENCE_YEARS = [2013, 2014, 2016, 2017, 2018, 2019, 2022]
SEASONAL_HARMONICS = 2          # annual + semi-annual
MIN_EVENT_MONTHS = 3            # observed (flag 0-1, C20) months of the 5 event months
MIN_REF_MONTHS = 36             # observed reference months
MIN_REF_YEARS = 4               # distinct reference years with an observed month
BOOT_B = 2000
BOOT_SEED = 20261014
CI_LEVEL = 0.90
MIN_SITES = 5                   # fewer qualifying sites -> 'not evaluable'
ZR_INFORMATIVE_M = 0.05         # RMS(M - ZR) anomaly below this -> M vs ZR not informative
PW_RADIUS_KM = (3.0, 10.0)      # distance strata to the nearest Yunlin Taiwan Water drought well
PW_WRA_TO_MODEL = 2.58          # WRA physical / model-unit volume ratio (plan P6 step 3)
ORIGIN = "2022-12"

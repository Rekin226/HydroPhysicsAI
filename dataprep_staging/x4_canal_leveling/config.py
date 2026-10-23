"""Fixed settings of WP X4 (canal below the district; public leveling results 2012-2025)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, L0_NEW, ORIGIN, REPO, REPORTS_DIR  # noqa: E402

WP = "X4"
FETCH_DATE = "2026-10-21"
UA = "Mozilla/5.0 (X11; Linux x86_64) HydroPhysicsAI-dataprep-X4 (research; polite, 1 req/s)"
MIN_PAUSE_S = 1.2

# ----------------------------------------------------------------------------- L0 sources
WRA_SITE = "https://web.wra.gov.tw/landsubsidence/water_new"
WRA_HOME = WRA_SITE + "/Home/IndexReNew"
WRA_LIST = WRA_SITE + "/RealtimeMonitor/GetDataReport/{series}"
WRA_FILE = WRA_SITE + "/File/ReportData/{series}/{name}"

L0_BRIEFINGS = L0_NEW / "wra_subsidence_briefings_007007" / FETCH_DATE
L0_SUMMARY = L0_NEW / "wra_subsidence_summary_007005" / FETCH_DATE
L0_CANAL = L0_NEW / "ia_irrigation_open_data" / FETCH_DATE

# in hand (read only)
L0_MISC = L0_NEW / "wra_reports_misc"
L0_CY007001 = L0_NEW / "wra_cy_monthly_reports_007001"
L0_RICE = L0_NEW / "moa_rice_J51_p3"
L0_RICE_PILOT = L0_NEW / "moa_rice_J51"
L0_CROP = L0_NEW / "moa_crop_038_p3"
L0_CROP_PILOT = L0_NEW / "moa_crop_038"
L0_IRR = L0_NEW / "moa_irrigation_stats"
LEVELING_PANEL = REPO / "chou-shui-data" / "data" / "ls_cache" / \
    "ls-wra-lsp-obs__choushui_panel.parquet"

# ----------------------------------------------------------------------------- products
PROD_LEVEL_POST = "subsidence_public_reports/postorigin"
PROD_LEVEL_PRE = "subsidence_public_reports/preorigin"
PROD_CANAL = "canal_redistribution_x4"
REPORT_DIR = REPORTS_DIR / "X4_canal_leveling"

COUNTIES = ("彰化縣", "雲林縣")

__all__ = ["DATAPREP", "L0_NEW", "ORIGIN", "REPO", "REPORTS_DIR"]

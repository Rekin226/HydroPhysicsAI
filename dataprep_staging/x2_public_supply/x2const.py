"""Constants of WP X2: paths, source ids and fixed assumptions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, L0_NEW, REPO, REPORTS_DIR  # noqa: E402

WP = "X2"
CODE_DIR = Path(__file__).resolve().parent
REPORT_DIR = REPORTS_DIR / "X2_public_supply"
EVID = REPORT_DIR / "evidence"
FETCH_LOG = REPORT_DIR / "fetch_log.jsonl"     # one line per HTTP request (all hosts)
API_CACHE = EVID / "api_cache"
TEXT_DIR = EVID / "text_layer"

FETCH_DATE = "2026-10-21"
MIN_INTERVAL_S = 1.5        # at least 1 s between requests (task rule 2); 1.5 s used
USER_AGENT = ("Mozilla/5.0 (research; HydroPhysicsAI data remediation; polite, "
              "1 request per 1.5 s)")

#: New L0 directories (each new and empty at first write; never an existing directory).
L0 = {
    "wra_water_use_annual": L0_NEW / "wra_water_use_annual" / FETCH_DATE,
    "twc_statistics": L0_NEW / "twc_statistics" / FETCH_DATE,
    "wra_subsidence_programme": L0_NEW / "wra_subsidence_programme" / FETCH_DATE,
    "datagovtw_twc_metadata": L0_NEW / "datagovtw_twc_metadata" / FETCH_DATE,
}

#: Existing L0 (read-only): the P6 sources and the WRA Changhua-Yunlin briefings.
L0_MISC = L0_NEW / "wra_reports_misc"
L0_PLANS = L0_NEW / "wra_subsidence_action_plans"

#: P6 member (read-only, through the guard).
P6_MEMBER = DATAPREP / "model_ready/release_20261012b2/preorigin/public_wells_member/v1"
P6_PRODUCT = DATAPREP / "public_wells/v2"
P6_MODEL_NPZ = DATAPREP / "public_wells/model_npz/v1"
ENGINE = DATAPREP / "changhua_abstraction/engine/v2"

ORIGIN = "2022-12"
WRA_TO_MODEL = 2.58          # P6's stated assumption (plan WP P6 step 3), kept for comparison
COUNTY_ZH = {"changhua": "彰化縣", "yunlin": "雲林縣"}

__all__ = ["REPO", "DATAPREP"]

# ------------------------------------------------------------------ fixed assumptions (X2)
#: Taiwan Water districts: 第十一區管理處 serves Changhua county only; Yunlin county is served by
#: 第五區管理處 (with Chiayi), through 林內/古坑/崙背/麥寮/四湖 systems (to 2015) and the
#: single 雲林供水系統 (from 2016) - verified in the yearbooks' 表1 service areas.
YUNLIN_SYSTEMS = ("林內供水系統", "古坑供水系統", "崙背供水系統", "麥寮供水系統", "四湖供水系統",
                  "雲林供水系統")
CHANGHUA_SYSTEMS = ("彰化供水系統", "花壇供水系統", "北斗供水系統", "二林供水系統",
                    "鹿港供水系統", "員林供水系統", "二水供水系統", "溪湖供水系統")
#: Whole townships each Changhua system serves (yearbook 102 = 2013, 表1 service areas:
#: townships printed with 全鄉/全鎮/全市 or 等全市鎮鄉; partial village lists dropped, except
#: 福興鄉, whose villages are split between the 鹿港 and 溪湖 systems). Wells are assumed to
#: lie in the service area (variant 'service_area' only).
SERVICE_TOWNS = {
    "彰化供水系統": ("彰化市", "和美鎮", "伸港鄉", "線西鄉"),
    "花壇供水系統": ("花壇鄉", "大村鄉"),
    "北斗供水系統": ("北斗鎮", "永靖鄉", "溪州鄉", "田尾鄉", "埤頭鄉", "埔心鄉"),
    "二林供水系統": ("大城鄉", "芳苑鄉", "竹塘鄉", "二林鎮"),
    "鹿港供水系統": ("鹿港鎮", "福興鄉"),
    "員林供水系統": ("員林市",),
    "二水供水系統": ("二水鄉", "田中鎮", "社頭鄉"),
    "溪湖供水系統": ("溪湖鎮", "埔鹽鄉", "秀水鄉", "福興鄉"),
}
TOWNS_TOPO = L0_NEW / "taiwan_atlas" / "towns-10t_2021.json"
#: Layers: the P6 member's convention for public deep wells (aquifers 3-4, equal split).
LAYER_WEIGHTS = (0.0, 0.0, 0.5, 0.5)
#: Drought years with extra public-well pumping, whose Jan-May share is not a normal year's:
#: 2021 and 2023 (WRA 112 briefing p. 26: '惟110及112年因大旱而增抽'; 2021 also in the 110
#: briefing p. 19 and the 110 drought-well survey), 2022 (drought-well survey 111: wells run
#: through 2022-12 for the southern drought from 2022-08).
DROUGHT_YEARS = (2021, 2022, 2023)
#: Uncertainty components (Mm3/yr unless stated).
SD_PLANT_FRAC = 0.03        # plant output vs abstraction (treatment and backwash losses)
SD_RIVERBED_2021 = 3.5      # 烏溪 riverbed intake listed with the wells from 2021 (彰化 system
#                             rose by 3.5 Mm3 from 2020 to 2021)
SD_UNKNOWN_PURCHASE = 1.4   # purchased raw water before 2012 (not parsed; 2012 value)
FIRST_MONTH, LAST_MONTH = "2007-01", "2025-12"

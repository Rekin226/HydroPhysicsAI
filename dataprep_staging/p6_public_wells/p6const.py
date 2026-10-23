"""Constants of WP P6: sources, paths and the sensitivity member's fixed assumptions.

Every assumption of the member is fixed here, before the member is built, and declared in
``P6.inventory_member.v1`` (harness.prereg3). None is chosen on a head, ring, pile, GNSS or
InSAR observation (U7): P6 reads no observation of those kinds at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, L0_NEW, REPO, REPORTS_DIR  # noqa: E402

WP = "P6"
PRODUCT = "public_wells"
CODE_DIR = Path(__file__).resolve().parent
EVID = REPORTS_DIR / "P6_evidence"
API_CACHE = EVID / "api_cache"
TEXT_DIR = EVID / "text_layer"          # pdftotext -layout output of the L0 PDFs (derived)
PREREG_ID = "P6.inventory_member.v1"

# ------------------------------------------------------------------ L0 sources (read-only)
L0_MISC = L0_NEW / "wra_reports_misc"
L0_PLANS = L0_NEW / "wra_subsidence_action_plans"      # fetched by fetch_plans.py (WP P6)
SRC = {
    "moea_annex": L0_MISC / "moea2019_yunchang_plan_annex.pdf",
    "cy53001": L0_MISC / "cy53001.pdf",
    "drought_wells": L0_MISC / "wra111_drought_wells_report.pdf",
    "brief108": L0_MISC / "wra_cy_subsidence_results_108.pdf",
    "brief109": L0_MISC / "wra_cy_subsidence_results_109.pdf",
    "brief110": L0_MISC / "wra_cy_subsidence_results_110.pdf",
    "brief111": L0_MISC / "wra_cy_subsidence_results_111.pdf",
    "brief112": L0_MISC / "wra_cy_subsidence_results_112.pdf",
    "plan_rev1": L0_PLANS / "moea_yunchang_plan_rev1_main.pdf",
    "plan_phase2": L0_PLANS / "moea_yunchang_plan_phase2_main.pdf",
    "plan_phase2_rev1_annex": L0_PLANS / "moea_yunchang_plan_phase2_rev1_annex.pdf",
}
SOURCE_JSON = [L0_MISC / "SOURCE.json", L0_PLANS / "SOURCE.json"]

# Publication vintage (YYYY-MM) of each source. A source published after the origin is
# post-origin even when it describes earlier years (U3): only the drought-well report is
# used in the pre-origin product, as a declared exemption (location proxy).
VINTAGE = {
    "cy53001": "2015-01",               # Control Yuan correction case 104財正0001 (2015)
    "moea_annex": "2019-10",            # 第二次修正 附冊, printed 108年10月 (2019-10)
    "plan_rev1": "2020-08",             # first revision, server Last-Modified 2020-08-24
    "plan_phase2": "2020-12",           # phase 2 (110-115), server Last-Modified 2020-12-07
    "brief110": "2022-06",              # WRA 110 (survey year 2021) results; month not
                                        # printed: assumed published before 2022-12
    "drought_wells": "2023-04",         # WRA 111 drought-well report (published 2023-04)
    "brief112": "2023-12",
    "plan_phase2_rev1_annex": "2026-06",
}

# Pages (1-based PDF page numbers) of the tables P6 reads.
PAGES = {
    "annex_t1_1": ("moea_annex", 4),           # 表1-1 and the public deep-well paragraph
    "p2_t4_5": ("plan_phase2", 47),            # 表4-5 phase-1 controlled public wells
    "p2_t4_5_text": ("plan_phase2", 48),       # (1)/(2)/(3) paragraphs after 表4-5
    "p2_at2_2": ("plan_phase2", 99),           # 附表2-2 counts and volumes by year
    "p2_at2_3": ("plan_phase2", 100),          # 附表2-3 achievement by agency
    "p2_at2_5": ("plan_phase2", 101),          # 附表2-5 irrigation wells by year
    "p2_summary": ("plan_phase2", 98),         # 附件2 text: 1,056 wells, 13,098.45
    "rev1_supply": ("plan_rev1", None),        # 公共用水 baseline paragraph (located by text)
    "b110_irr_yl": ("brief110", 18),           # 灌溉抽水量比較, Yunlin (ratio check)
}

# Claude API (decision D6; P6 approved for transcription only, never to fill values)
MODEL_PDF = "claude-opus-5-5"        # first read: the page as a one-page PDF document
MODEL_IMG = "claude-sonnet-5-5"      # second, independent read: the page rendered as PNG
RENDER_DPI = 200

# ------------------------------------------------------------------ census (C0 products)
C0_L1 = DATAPREP / "pump_electricity/census_L1/v3"
C0_PRE = DATAPREP / "pump_electricity/preorigin/v3"
ENGINE = DATAPREP / "changhua_abstraction/engine/v2"
N1_PRODUCT = DATAPREP / "natural_experiments/v1"
GRID_NPZ = REPO / "results/twin_forward/datum_gate_innov.npz"
COUNTIES = ("yunlin", "changhua", "chiayi")
COUNTY_ZH = {"yunlin": "雲林縣", "changhua": "彰化縣", "chiayi": "嘉義縣"}
# census purposes that name public supply (公共給水 / 公共用水), the proxy of plan P6 step 3
PUBLIC_PURPOSES = ("公共用水", "家用及公共給水", "家用及公共用水", "家庭及公共給水",
                   "民生用水-公共")
INDUSTRIAL_PURPOSES = ("工業用水",)
CENSUS_ENERGY_YEARS = (2012, 2021)    # census energy summaries (avoids the 2022-12 fill)
CENSUS_MATCH_M = (50.0, 200.0)        # drought-well to nearest census pump distance strata

# ------------------------------------------------------------------ member (plan P6 step 3)
ORIGIN = "2022-12"
MEMBER_T0 = "2007-01"                 # time index 0, the C0/P2 convention
MEMBER_T1 = "2022-12"                 # pre-origin member ends at the origin
DISPOSAL_MONTH = 7                    # a cohort of year Y stops pumping from Y-07 (mid-year)
TIMING_VARIANTS = {"timing_early": 1, "timing_late": 13}   # Y-01, (Y+1)-01
DAYS_PER_YEAR = 365.25                # 萬噸/年 -> m3/day: v * 1e4 / 365.25 (constant rate)
WRA_TO_MODEL = 2.58                   # WRA physical / model-unit volume (plan P6 step 3)
MEMBER_AQUIFERS = (3, 4)              # plan P6 step 3: layers 3-4 = aquifers 3-4
MEMBER_AQUIFER_SPLIT = (0.5, 0.5)     # equal split, a stated assumption (no screen data)
N_LAYERS = 4                          # twin layer index = aquifer code - 1
IA_VARIANT_AQUIFER = 2                # Control Yuan 53001: IA wells 120-150 m (aquifer 2)

# Columns of 附表2-2 (left to right) -> cohort keys. 'proxy' names the location proxy.
COHORT_COLUMNS = [
    {"key": "tw_changhua", "county": "changhua", "agency": "taiwan_water",
     "label": "自來水公司(11區)", "proxy": "drought:Taiwan Water Corporation (台水)"},
    {"key": "ia_changhua", "county": "changhua", "agency": "irrigation_association",
     "label": "彰化農田水利會", "proxy": "drought:Irrigation Agency (農水署)"},
    {"key": "ts_changhua", "county": "changhua", "agency": "taiwan_sugar",
     "label": "台糖公司", "proxy": "census_public"},
    {"key": "factory_changhua", "county": "changhua", "agency": "factory",
     "label": "溪州地區工廠", "proxy": "census_industrial_town:溪州", "town": "溪州"},
    {"key": "school_changhua", "county": "changhua", "agency": "school",
     "label": "下陷區國中小學", "proxy": "census_public"},
    {"key": "tw_yunlin", "county": "yunlin", "agency": "taiwan_water",
     "label": "自來水公司(5區)", "proxy": "drought:Taiwan Water Corporation (台水)"},
    {"key": "ia_yunlin", "county": "yunlin", "agency": "irrigation_association",
     "label": "雲林農田水利會", "proxy": "drought:Irrigation Agency (農水署)"},
    {"key": "ts_yunlin", "county": "yunlin", "agency": "taiwan_sugar",
     "label": "台糖公司", "proxy": "drought:Taiwan Sugar (台糖)"},
    {"key": "factory_yunlin", "county": "yunlin", "agency": "factory",
     "label": "元長地區工廠", "proxy": "census_industrial_town:元長", "town": "元長"},
    {"key": "school_yunlin", "county": "yunlin", "agency": "school",
     "label": "下陷區國中小學", "proxy": "census_public"},
    {"key": "prison_yunlin", "county": "yunlin", "agency": "prison",
     "label": "雲林第二監獄", "proxy": "census_public"},
]
YEARS_ROC = list(range(100, 109))      # 附表2-2 rows 100-108 = 2011-2019
ANNEX_YEARS_ROC = list(range(100, 108))   # the 2019 annex window 100-107 = 2011-2018

# Annex totals (MOEA 2019 annex p. 2, 表1-1 and paragraph 4) and the plan's acceptance.
ANNEX_WELLS = 774
ANNEX_VOLUME_1E4 = 8793.9
RECONCILE_TOL = 0.05                   # plan WP P6 acceptance: within 5 %

# Pending Changhua Taiwan Water wells (表4-5: 129 wells planned 109-113 with Niaozuitan;
# 附表2-3: full target 7,172.25 vs 1,842.3 done by 108). A separate component, not the ramp.
PENDING_TW_CH_WELLS = 129

VARIANTS = ("primary", "annex_2011_2018", "timing_early", "timing_late", "layers_ia_aq2",
            "loc_census_only", "loc_drought_only", "loc_uniform_county")

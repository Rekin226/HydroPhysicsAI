"""Constants of WP S1b, fixed in the pre-registration S1b.ring_bridge_checks.v1.

Every number is the plan's (section 3, WP S1, checks (c) and (d)), an S1a or S4b convention
reused unchanged, or a choice declared before any S1b statistic was computed on real data.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATAPREP = REPO / "results" / "dataprep"
RINGS_V2 = DATAPREP / "rings_bridged" / "v2"
GNSS_FULL_V2 = DATAPREP / "gnss_clean" / "full" / "v2"
INSAR_V2 = DATAPREP / "insar_069A" / "v2"
RING_STATIONS = REPO / "chou-shui-data" / "data" / "mlcw_stations.csv"   # site coordinates
TWIN_GRID = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
EVIDENCE = DATAPREP / "reports" / "S1b_evidence"
WP = "S1b"
PREREG_ID = "S1b.ring_bridge_checks.v1"

#: The 14 ring sites (S1a, plan WP S1).
SITES = ("僑義國小", "元長國小", "光復國小", "內寮派駐站", "北辰國小", "嘉興國小", "土庫國中",
         "宏崙國小", "客厝國小", "拯民國小", "新生國小", "湖南國小", "溪州國小", "秀潭國小")
SITE_EN = {"僑義國小": "Qiaoyi", "元長國小": "Yuanchang", "光復國小": "Guangfu",
           "內寮派駐站": "Neiliao", "北辰國小": "Beichen", "嘉興國小": "Jiaxing",
           "土庫國中": "Tuku", "宏崙國小": "Honglun", "客厝國小": "Kecuo", "拯民國小": "Zhengmin",
           "新生國小": "Xinsheng", "湖南國小": "Hunan", "溪州國小": "Xizhou", "秀潭國小": "Xiutan"}

# ----------------------------------------------------------------------------- pairing
#: A GNSS station is co-located with a ring site when it lies within this distance (the WRA
#: stations sit at the ring schools, 0-150 m).
COLOCATED_KM = 0.3
#: NGL stations "near ring sites" (plan (c): "to be checked from coordinates"): within 5 km.
NGL_NEAR_KM = 5.0
#: Report only: for a site without a co-located WRA station, its nearest station of either
#: family within 6 km that has pre-gap data.
NEIGHBOUR_KM = 6.0
#: A station needs data before this day to have a pre-gap window.
PRE_GAP_END = "2021-12-31"

# ----------------------------------------------------------------------------- windows
#: Pre-gap window: rates and the seasonal harmonic of every instrument and of the ring.
PRE_WIN = ("2020-01-01", "2021-12-31")
#: Post-gap window (rates only, seasonal held at the pre-window harmonic). Both ends are
#: clipped to the instrument's data and to the ring's first post-gap survey (Beichen
#: 2024-07-11); the ring is fitted on exactly the same clipped window.
POST_WIN = ("2023-02-01", "2025-05-31")
#: (Set from availability only, in the synthetic run before registration: 270 d left the
#: Beichen-SLES post window (263 d, GNSS ends 2025-03) unevaluable.)
MIN_SPAN_DAYS = 240            # first to last sample inside a window
MIN_GNSS_DAYS = 240            # valid daily heights per window
MIN_INSAR_EPOCHS = 20          # valid epochs per window
MIN_RING_SURVEYS = 8           # surveys with both rings per window

#: GNSS flags used (gnss_clean: 0 observed, 1 corrected (metadata/catalogue/own series),
#: 3 corrected (common-mode-only step); 9 suspect excluded).
GNSS_FLAGS = (0, 1, 3)
GNSS_END_HALF_D = 3            # endpoint: mean of daily heights within +-3 d (S1a, pilot)
GNSS_END_HALF_D_FALLBACK = 7   # ... or +-7 d when +-3 d holds fewer than 3
GNSS_END_MIN = 3

#: InSAR: S4b's station rule (median of the 3 x 3 multilooked pixels around the nearest
#: pixel, >= 4 valid in the LiCSBAS15 mask), restricted to pixels whose segment ids equal the
#: network's at every epoch (no pixel-specific gap).
INSAR_PIX_HALF = 1
INSAR_PIX_MIN_VALID = 4
INSAR_END_HALF_D = 18          # endpoint: mean of the epochs within +-18 d ...
#: ... at least 1, all in the segment of the nearest epoch (availability-only choice before
#: registration: Beichen's first post-gap survey, 2024-07-11, has a single epoch (2024-07-01)
#: of its segment within 18 d; the other 13 endpoints have 3-6 epochs; noise scales as 1/n).
INSAR_END_MIN = 1
#: Declared InSAR noise (uncalibrated): per-epoch sd 6.6 mm (S4b's pre-origin run-P
#: within-segment single-interval InSAR-GNSS increment RMS 9.4 mm split over two epochs; GNSS
#: noise is part of it, so this is conservative), and 6 mm per network-gap increment that
#: LiCSBAS bridged with its linear-velocity constraint (a priori: the deviation of a 24-36 d
#: summer increment from a linear-velocity bridge for a 10-20 mm seasonal amplitude).
INSAR_EPOCH_SD_MM = 6.6
INSAR_BRIDGE_SD_MM = 6.0

# ----------------------------------------------------------------------------- tests
TOL_SIGMA = 2.0                # bracket widened by 2 sigma on each side
FAMILY_MIN_PASS_FRAC = 0.8     # (c) per family and (d2): >= 80 % of evaluable sites contain
FAMILY_MIN_SITES_GATE = 3      # a family enters the S1b gate only with >= 3 evaluable sites
CORR_ALPHA = 0.05              # (d1): one-sided 5 % critical r for the evaluable n
SEED = 1802                    # synthetic operating characteristics (design-time)
SYN_REAL = 200                 # realizations per site and scenario

#: S1a reference scales for the report-only evidence lines (s1const.REF).
REF_PPM = {"A": (182.2, 11.4), "B": (1.9, 14.3)}

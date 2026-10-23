"""Constants of WP S1a, fixed in the declaration S1a.rings_bridged_rules.v1 (plan WP S1).

Every number here is either the plan's (section 3, WP S1) or a choice declared before the
rebuilt evidence was computed; the declaration file records them verbatim.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPACTION_DIR = REPO / "data_fetch_api" / "new_measurements" / "compaction"
DEEP_BENCH_V2 = REPO / "results" / "dataprep" / "deep_benchmarks" / "v2"
PRINTED_V2 = REPO / "results" / "dataprep" / "deep_benchmarks_printed" / "v2"
GNSS_FULL_V1 = REPO / "results" / "dataprep" / "gnss_clean" / "full" / "v1"
EVIDENCE = REPO / "results" / "dataprep" / "reports" / "S1a_evidence"
PRODUCT = "rings_bridged"

#: The 14 ring sites surveyed on both sides of the 2022 hiatus (plan WP S1).
SITES = ("僑義國小", "元長國小", "光復國小", "內寮派駐站", "北辰國小", "嘉興國小", "土庫國中",
         "宏崙國小", "客厝國小", "拯民國小", "新生國小", "湖南國小", "溪州國小", "秀潭國小")
BEICHEN = "北辰國小"

# ----------------------------------------------------------------------------- windows
PRE_WIN = ("2019-01-01", "2021-12-31")           # window fits before the gap
POST_WIN = ("2023-02-01", "2025-05-31")          # after it
POST_WIN_BEICHEN = ("2024-07-01", "2025-05-31")  # Beichen resumed in July 2024
GAP_LAST_PRE = "2021-12-31"                       # no survey 2022-01..2023-01 at any site
GAP_FIRST_POST = "2023-01-01"
CLIM_WIN = ("2014-05-01", "2021-12-31")          # seasonal climatology (after the 2012-14 hiatus)
PILE_PRE_WIN = ("2017-01-01", "2021-12-31")      # ring/pile rate ratio before the gap
PILE_POST_WIN = ("2023-02-01", "2025-05-31")     # and after it
COVER = 0.8            # a ring is used if read in >= 80 % of the surveys of both windows
RATE_THR = 1e-3        # m/yr: "low-rate" deep interval (pilot method C)
MIN_DEEP = 3           # deep intervals needed for a per-site scale

# ----------------------------------------------------------------------------- evidence
#: Reference group scales used only to weigh evidence lines (plan WP S1 pilot values):
#: (mean, between-site spread) in strain units.
REF = {"A": (182.2e-6, 11.4e-6), "B": (1.9e-6, 14.3e-6)}
SD_FLOOR = 10e-6       # floor on a per-site scale sd (pilot pooling floor)
#: Offset clusters of the top-ring jump (m): the plan's ranges -15.5..-16.3 cm and +50..+54 cm,
#: each widened by 0.5 cm (about two standard errors of a window-fit jump).
OFFSET_CLUSTERS = {"A": (-0.168, -0.150), "B": (0.495, 0.545)}
OFFSET_LR = 10.0       # likelihood ratio carried by an offset in a group's cluster
LR_VOTE = 10.0         # a continuous line votes when its likelihood ratio reaches 10 (or 1/10)
Z_PLAUSIBLE = 3.0      # ... and its value lies within 3 sd of the favoured group
LR_CAP = 1.0e3         # each line's likelihood ratio is capped at 1e3 (and 1e-3)
PRIOR_A = 0.5          # prior membership probability
#: Carried as both members in any case (plan WP S1: near-threshold or uncorroborated).
FORCED_BOTH = frozenset({"內寮派駐站", "元長國小", "秀潭國小", "僑義國小"})
#: Synthetic analogue of the near-threshold criterion: |dS_obs| <= 1.0 cm.
NEAR_THRESHOLD_M = 0.010

# ----------------------------------------------------------------------------- piles
#: site -> (sensor, pile depth m, use). "total": pile-shaped 2022 path of the total interval
#: and a pile-implied scale; "shallow": the band from the top ring to the ring nearest the
#: pile depth is pile-shaped (Xiutan, plan); "evidence": pile-implied scale only.
PILES = {
    "土庫國中": ("TKJS(1)", 300.0, "total"),
    "溪州國小": ("SJES", 300.0, "total"),
    "光復國小": ("NTUH(1)", 300.0, "total"),
    "秀潭國小": ("STES", 70.0, "shallow"),
    "拯民國小": ("NTUH(1)", 300.0, "evidence"),
}
PILE_TOL_DAYS = 3      # a pile value at a survey date: daily values within +-3 days
PILE_FILL_LIMIT = 31   # the 2022 path may interpolate pile gaps of up to 31 days
PILE_WIN_MONTHS = 14   # representativeness term: ring vs pile changes over 14-month spans
PILE_MIN_COVER = 0.8   # daily coverage needed in each rate window

# ----------------------------------------------------------------------------- path
BAND_EDGES_M = (0.0, 100.0, 200.0)   # as hydrophysics.twin.mechanism.BAND_EDGES_M
PATH_YEARS = (2017, 2018, 2019, 2020)  # family Y, pre-origin (2021 has no January-2022 survey)
TESTED_GAP_MONTHS = 13                 # the path test spans December -> January (13 months)

# ----------------------------------------------------------------------------- tests
WORST_WINDOW_RMSE_M = 0.015    # (e) post hoc threshold (plan): labelled post hoc wherever quoted
POOLED_Z_MAX = 2.0             # (a)
POWER_MIN = 0.8                # (a) power for a 25 % scale error
SCALE_ERROR = 0.25
PILE_REL_TOL_A = 0.25          # (b) group A within 25 % of the pooled scale
PILE_ABS_TOL_B = 30e-6         # (b) group B within 30 ppm
RESID_OFFSET_M = 0.010         # residual offset within +-1 cm ...
RESID_OFFSET_MIN_SITES = 12    # ... at >= 12 of 14 sites
SYN_WINDOW = ("2015-01", "2021-12")
SYN_LEN = 13
SYN_PRE_MONTHS = 36
SYN_POST_MONTHS = 28
SYN_MIN_PRE = 20               # surveys
SYN_MIN_POST = 18
SYN_MIN_SITES = 10
SYN_REAL = 10                  # realizations per window; seed rule 1000*ord('R') + 100*w + k
OFFSET_OTHER = (-0.10, 0.20)   # harness default offset range outside the clusters (m)

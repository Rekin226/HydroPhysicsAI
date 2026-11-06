"""PB.decomposed_band.v1 (round-8 rank 16, PB part; user ruling R8.13): the constants that the
declaration draft (prereg_pb.py) fixes and the band builder (build_pb.py) obeys.

R8.13 (user_rulings_2026-11-03.json, sha ec610b69) replaces the drawn aggregate run spread of
the page's head series with a two-part band:

1. a per-well ZONE RIBBON from E02d's frozen relative half-widths (sigma_idio_A/B/C.csv,
   registered by hash in E02d.decomposed_prospective_2026-12.v1): per DR zone (E02d's frozen
   ``group``) and month, the median over the zone's wells of half_width_m = z90 sigma_idio;
   a zone with fewer than 5 wells is not drawn; no station id reaches the page. Label:
   'decided by a registered test, E02d, 2028' (window A; windows B and C carry E02y's
   registered wording);
2. the SHARED RANGE (the county-wide part E02d removed), per county and month of window A:
   the hull of {0 (E02's registered centre), the 36 rerun members' county-median deviation
   from the member mean, the four FV v2 truth families' county-median departure from E02's
   registered forward}. Label: 'cannot be tested before 2030 by design'.

Beside the shared range PB publishes its leave-one-family-out (LOFO) coverage per county:
design-time and model-only (no observation), on window A's months only. Leave-one-forcing-
year-out is dropped (plan #16) and never computed. The U7 guard (no 2023-26 coverage of the
relative recipe, and no coverage of any kind on a month before 2026-12) lives in build_pb.py
and tests/test_pb_band.py only, never in hydrophysics/twin/coverage.py, bands.py or
harness/coverage3.py.

Nothing here reads a file; every hash below was recorded by an upstream registration (E02d's
interval.artefact.files and inputs, E02d's summary.json inputs, the INDEX rows) and is
cross-checked against those documents by prereg_pb.problems before registration.
"""
from __future__ import annotations

VERSION = "1.0.0"
PB_ID = "PB.decomposed_band.v1"
WP = "PB"
X = "dataprep_staging/pb_band"
SELF = f"{X}/pb_const.py"
BUILDER = f"{X}/build_pb.py"
DRAFT = f"{X}/prereg_pb.py"
TESTS = "tests/test_pb_band.py"
#: PB's own code, frozen at registration (the builder refuses if any changed)
CODE_FILES = (SELF, BUILDER)
#: upstream modules the builder imports (functions used unchanged); frozen too
UPSTREAM_CODE = ("dataprep_staging/e02d/e02dlib.py", "dataprep_staging/e02d/e02dio.py",
                 "dataprep_staging/e02d/power_e02d.py",
                 "dataprep_staging/fb_family/fb_inputs.py",
                 "dataprep_staging/fb_family/fb_lib.py", "dataprep_staging/fb_family/fb_io.py",
                 "dataprep_staging/fv_decidability/osse_fv.py")
FROZEN = CODE_FILES + UPSTREAM_CODE

# ----------------------------------------------------------------------------- registrations
PREREG = "results/dataprep/preregistration"
E02D_ID = "E02d.decomposed_prospective_2026-12.v1"
E02D_SHA = "3dcdb4689a846c19c6e80d9af3112ada7437255627400adddbc67362555adb01"
E02Y_ID = "E02y.years23_2027-11_2029-10.v1"
E02Y_SHA = "4bf7d91f928c8017176c5f53a60abffe5f86a9cc93386d521e5fd8bc7e0c6d12"
FV2_ID = "FV.decidability.v2_postfailure"
FV2_SHA = "f7e35a17482cce307c6be84013f9a2919f16f4be66e9a01cd7eeacbd08fb49d4"
DR_ID = "DR.relative_drift.v1"
DR_SHA = "f835af572a8b6e49bec6ae38d6edf7c37bb9d8e26abedb658b71e6e9c5fe873b"
FB_ID = "FB.family_power.v1"
FB_SHA = "bc8a935a70ebdda1955c33f359a463a2a92dd08f6a55459e31d3130fc7bc61a8"
FB_RESULT = f"results/dataprep/reports/{FB_ID}/result.json"
FB_RESULT_SHA = "bdc0589e6f58c44d46e03d72b55c89d6558f357013a6f9f7cb0279509fdfb4a2"
U1X_ID = "U1.ext_transfer_power.v1"
U1X_SHA = "1bfbf0100965dcaef24f5242b051072373280aecdd057e650c43ad59c9f7495b"
U1X_RESULT = f"results/dataprep/reports/{U1X_ID}/result.json"
U1X_RESULT_SHA = "daace1bf045433ca68576118a0dd58b27c51f10f24622f1e9b5c7095500affb8"
E02X_ID = "E02x.ext_transfer_2026-12.v1"
UPSTREAM_REGS = {E02D_ID: E02D_SHA, E02Y_ID: E02Y_SHA, FV2_ID: FV2_SHA, DR_ID: DR_SHA}

RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.13"
RULING_DECISION = ("Yes: per-well zone ribbon 'decided by a registered test, E02d, 2028' (min 5 "
                   "wells per zone, no station ids) plus the shared range 'cannot be tested "
                   "before 2030 by design'.")

# ----------------------------------------------------------------------------- inputs
#: files under results/twin are read from the builder's --twin-dir (a checkout's
#: results/twin may be partial); they are bound by hash, never by location
TWIN_PREFIX = "results/twin/"
SIGMA = {"A": "results/twin/e02d_relative/sigma_idio_A.csv",
         "B": "results/twin/e02d_relative/sigma_idio_B.csv",
         "C": "results/twin/e02d_relative/sigma_idio_C.csv"}
#: E02d interval.artefact.files (registered)
SIGMA_SHA = {"A": "b70da8b0f2d340d9609e7f7423ab8961596aa21292b1f578c3bfbf89acd51988",
             "B": "aa3d969528a5f775af720844cd63d936094e8aac437cefd31e552425fd47be90",
             "C": "29ce843d1e7c96ab278e8bb31d864b2dbc5b77d44b2400708c8347d65e947fe5"}
SUMMARY = "results/twin/e02d_relative/summary.json"
SUMMARY_SHA = "7e23aee9940cb4fa1be2d1f61403282302892fa6a1ff1531baa3422b295a1909"
#: the centre E02d's artefact carries: E02's registered predictions (A) and the same recipe's
#: absolute registrations (B, C); E02d summary.json inputs abs_A/B/C
CENTRE = {"A": "results/twin/prospective_2026-11/predictions.csv",
          "B": "results/twin/prospective_2027-11_y2/predictions.csv",
          "C": "results/twin/prospective_2028-11_y3/predictions.csv"}
CENTRE_SHA = {"A": "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
              "B": "49e23b7e7f667f7aef7a64c4cbb5228863a2d1c7631b812f9c00516ee5930653",
              "C": "c494fa040e487b6912edd30a2facc99746e853125e7e61d4c3f94fbd4f473eaa"}
WELLS_E02D = "results/dataprep/reports/DR.relative_drift.v1/wells_e02d.csv"
WELLS_E02D_SHA = "f3174c481827f22efb4a6c7436359e765414bff07fa9d8bd06a65a73c51a92b8"
MEMBERS_NPZ = "results/twin_runs/e02d/members_wells.npz"
MEMBERS_SHA = "12c32ff45500f4c6827a6ea60b9e2e5e39b19c94835faed290f88db4d6292262"
#: the family part reads FV v1's truth forwards, E02's forward, wells_158 and obs_mask through
#: fb_inputs.load_mu (paths and FB-recorded sha256 in fb_inputs.UPSTREAM_SHA)
FAMILY_INPUT_KEYS = ("wells_158", "obs_mask", "zr", "truth_R0", "truth_Q0", "truth_R1c",
                     "truth_X5R1")
#: path fragments of observation products: no PB input may contain one (U7 / no heads)
OBSERVATION_MARKERS = ("AMP_V2/data", "chou-shui-data", "data_fetch_api",
                       "results/dataprep/heads", "fan_stations", "step5", "new_wells",
                       "window_product", "heads_window")

# ----------------------------------------------------------------------------- design
WINDOWS = {"A": ("2026-12", "2027-10"), "B": ("2027-11", "2028-10"),
           "C": ("2028-11", "2029-10")}
WINDOW_MONTHS_N = {"A": 11, "B": 12, "C": 12}
#: the registered test that decides the relative band of each window
WINDOW_TEST = {"A": {"registration": E02D_ID, "test_id": "E02dP.G_idio_2026-12",
                     "not_before": "2028-01-15"},
               "B": {"registration": E02Y_ID, "test_id": "E02yP.rel_2027-11",
                     "not_before": "2029-01-15"},
               "C": {"registration": E02Y_ID, "test_id": "E02yP.rel_2028-11",
                     "not_before": "2030-01-15"}}
ORIGIN = "2022-12"
Z90 = 1.6448536269514722
TOL_M = 1e-9
MIN_ZONE_WELLS = 5
COUNTIES = ("Yunlin", "Changhua")
COUNTY_CODE = {"07": "Changhua", "09": "Yunlin"}
N_WELLS = 158
WELLS_BY_COUNTY = {"Changhua": 74, "Yunlin": 84}
N_MEMBERS = 36
#: FV v2's truth families (recipe ids only; FV.decidability.v2_postfailure truths[].tag)
FAMILIES = ("R0", "Q0", "R1c", "X5R1")
#: FB's two-family partition (FB.family_power.v1 hypotheses.families); report-only LOFO
FB_GROUPS = {"R0Q0": ("R0", "Q0"), "R1cX5R1": ("R1c", "X5R1")}
#: the shared range (and its LOFO coverage) exists where both parts exist: window A
SHARED_WINDOW = "A"
#: U7: months whose coverage was seen when the interval family was chosen
U7_SEALED = ("2023-01", "2026-11")
FIRST_COVERAGE_MONTH = "2026-12"
DROPPED = {"leave_one_forcing_year_out": "dropped (round-8 plan #16): not computed and not "
                                         "published"}
ROUND = 4                                       # metres, 4 decimals in the output

# ----------------------------------------------------------------------------- page wording
LABELS = {
    "ribbon_A": {"en": "decided by a registered test, E02d, 2028",
                 "zh": "由已登錄之檢定 E02d 於 2028 年判定"},
    "ribbon_BC": {"en": "outside the first window, decided by registered tests for "
                        "2027-11..2029-10 (E02y); untested beyond",
                  "zh": "第一個檢驗期之外，由 2027-11..2029-10 的已登錄檢定（E02y）判定；"
                        "其後未經檢驗"},
    "ribbon_what": {"en": "Per-well spread around the zone's common departure (median over "
                          "the zone's wells of E02d's frozen 90 % relative half-width); not "
                          "the level, not the policy contrast, not the full interval",
                    "zh": "各井相對於分區共同偏移的離散（E02d 凍結之 90 % 相對半寬，分區內各井"
                          "之中位數）；不是水位本身、不是政策對比、也不是完整區間"},
    "ribbon_small": {"en": "zones with fewer than 5 wells are not drawn",
                     "zh": "井數少於 5 口的分區不繪製"},
    "shared": {"en": "cannot be tested before 2030 by design",
               "zh": "依設計，2030 年前無法檢定"},
    "shared_what": {"en": "Shared range, common to every well of the county: the hull of the "
                          "frozen forecast, its 36 runs' county-wide deviation and the four "
                          "recipe families' county-wide departure (model-only; not a "
                          "predictive interval)",
                    "zh": "全縣各井共有的範圍：凍結預測、其 36 次模擬之全縣偏移與四個配方族群之"
                          "全縣偏移所圍成的範圍（僅模型；不是預測區間）"},
    "shared_end": {"en": "not drawn after 2027-10 (the family forwards end there)",
                   "zh": "2027-10 之後不繪製（族群模擬至此為止）"},
    "lofo": {"en": "leave-one-family-out coverage (design-time, model-only): {county} {k} of "
                   "{n} family-months inside the range built without that family",
             "zh": "留一族群覆蓋（設計階段，僅模型）：{county} {n} 個族群月中有 {k} 個落在不含"
                   "該族群所建的範圍內"},
}
NEVER = ("calibrated", "校準", "the full interval passed", "predictive interval passed")

# ----------------------------------------------------------------------------- #16 status
FB_STATUS = ("FB.family_power.v1 (registered sha bc8a935a; result sha bdc0589e) failed its gate "
             "in both counties (worst-row P(correct strong) Changhua 0.21, Yunlin 0; threshold "
             "0.8): no BF test is registered; ledger row E29 'family BF not decidable by "
             "design'; no page line")
EXS_STATUS = ("EXS not run: its precondition (E02x registered) failed. U1.ext_transfer_power.v1 "
              "(registered sha 1bfbf010; result sha daace1bf) failed its P-gate, so "
              "E02x.ext_transfer_2026-12.v1 was never registered (row C14f-r N); nothing is "
              "built for EXS")

OUT_DIR = "results/twin/pb_band"
OUT_JSON = "pb_band.json"
ATTEMPTS = "results/twin_runs/pb_band/attempts.jsonl"

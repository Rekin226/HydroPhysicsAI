"""WP S4b (InSAR processing): fixed settings, paths and registered rules.

Plan section 3 "WP S4" steps 3-8 and its acceptance; user decisions D1, D11 = none, D12.
Everything a registered test depends on is declared here and copied into the
pre-registration ``S4b.insar_acceptance.v1`` (``s4b_prereg.py``). No study-area coordinates
live here: the clip windows, reference geotransforms, station positions and the twin grid are
read at run time from gitignored products (``insar_licsar_subsets/v1``, the twin grid npz,
the vendor GNSS metadata).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, REPO  # noqa: E402

WP = "S4b"
PREREG_ID = "S4b.insar_acceptance.v1"
PRODUCT = "insar_069A"

PRIMARY = "069A_06694_131402"
DESC = "105D_06628_232108"

#: S4a hand-over (L1 index product; gate passed on its data criteria).
S4A_PRODUCT = DATAPREP / "insar_licsar_subsets" / "v1"
S4A_NOTES = S4A_PRODUCT / "notes_for_S4b.json"
S4A_INVENTORY = S4A_PRODUCT / "inventory.json"
S4A_RUN = DATAPREP / "insar_tools" / "s4a_run"
TWIN_GRID = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
UNITS_V3 = DATAPREP / "validation_units" / "v3" / "units.json"

RUN_DIR = DATAPREP / "insar_tools" / "s4b_run"           # LiCSBAS work dirs, scratch (ignored)
EVID = DATAPREP / "reports" / "S4b_evidence"
LICSBAS_COMMIT = "a145e6c4217b29eb7348ea95630454a099809098"

#: LiCSBAS runs (s4b_run_licsbas.sh): name -> (frame, first, last yyyymmdd on both epochs).
RUNS = {
    "P": (PRIMARY, "20200101", "20221231"),   # pre-origin gate run (U7)
    "F": (PRIMARY, "20200101", "20251231"),   # product run; drops the 4 empty 20260104 pairs
    "D": (DESC, "20250101", "20260131"),      # descending, 2025 decomposition (after the fact)
}
NLOOK = 2
LICSBAS_STEPS = [
    "LiCSBAS02_ml_prep.py -i GEOC -o GEOCml2 -n 2 --n_para 1",
    "LiCSBAS11_check_unw.py -d GEOCml2 -t TS_GEOCml2",
    "LiCSBAS12_loop_closure.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --nopngs",
    "LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --mem_size 4000 --nopngs",
    "LiCSBAS14_vel_std.py -t TS_GEOCml2 --mem_size 4000",
    "LiCSBAS15_mask_ts.py -t TS_GEOCml2",
]

#: Stable reference area (plan S4 step 5: offsets and ramps from off-fan references only).
#: A multilooked pixel is stable when it is land, its LiCSAR DEM height lies in
#: [STABLE_MIN_HGT_M, STABLE_MAX_HGT_M] (bedrock tablelands and low hills, not alluvium; the
#: cap keeps the stratified-delay fit near the fan's own heights), its centre lies at least
#: STABLE_MIN_DIST_KM from every active twin-cell centre, and it passes the LiCSBAS15 mask.
STABLE_MIN_HGT_M = 100.0
STABLE_MAX_HGT_M = 600.0
STABLE_MIN_DIST_KM = 2.5
STABLE_MIN_PIXELS = 200          # an epoch whose stable set is smaller is flagged 9
#: Per-epoch reference model fitted on the stable pixels (robust: two passes, rejection at
#: 3 x 1.4826 x MAD of the first-pass residuals): offset + east/north plane + height term.
REF_TERMS = ("offset", "east_km", "north_km", "hgt_km")
REF_SENSITIVITY = ("offset",)    # reported: offset (median of stable pixels) only

#: Vertical-equivalent (plan S4 step 6): LOS (towards the satellite, LiCSBAS cum) / U, with
#: U the per-pixel up component of the LOS unit vector (frame U raster, 2 x 2 mean).
VERTICAL = "los_over_u_per_pixel"
U_CONSTANT_PLAN = 0.833          # plan's constant, reported as a sensitivity

#: Twin-cell aggregation of multilooked pixels (1 km cells): median of valid pixels whose
#: centre lies in the cell; at least CELL_MIN_PIXELS valid pixels, otherwise flag 8.
CELL_MIN_PIXELS = 5

#: GNSS leave-one-station-out test (plan S4 acceptance; pre-origin window for U7).
GNSS_T0 = "2021-09-30"           # centre of the first window
GNSS_T1 = "2022-11-30"           # centre of the second window (14 months later)
GNSS_HALF_WINDOW_D = 30
GNSS_LOSO_RMS_MAX_MM = 10.0
GNSS_MIN_STATIONS = 5            # fewer evaluable stations = test not evaluable (fail)
GNSS_STEP_RULE = "exclude"       # stations with a catalogued step in the span are left out;
#                                  correcting them by the catalogue size is report-only
GNSS_MIN_DAYS_PER_WINDOW = 20    # daily GNSS values inside each +-30 d window
GNSS_MIN_EPOCHS_PER_WINDOW = 3   # InSAR epochs with both values inside each window
GNSS_PIX_HALF = 1                # 3 x 3 multilooked pixels around the station
GNSS_PIX_MIN_VALID = 4
GNSS_SMOOTH_DAYS = 7             # centred rolling median of the daily heights
GNSS_MATCH_TOL_D = 3
#: Plan step catalogue (WP S3), pre-origin entries only (U7: 2023+ steps unused).
GNSS_STEP_CATALOGUE = {"TKJS": [("2022-06-17", 0.1323)], "SJES": [("2022-03-29", 0.100)],
                       "GFES": [("2020-07-10", -0.056)]}
#: The pilot's post-origin window (2021-12-05 -> 2023-02-20), reported after the fact only.
PILOT_T0, PILOT_T1 = "2021-12-05", "2023-02-20"

#: Changhua coherence coverage (plan S4 acceptance), pre-origin summer pairs.
COH_MIN = 0.30
COH_ZONE_MIN_FRAC = 0.30
COH_SUMMER_MONTHS = (6, 7, 8, 9)
COH_SUMMER_YEARS = (2020, 2021, 2022)
COH_MAX_BTEMP_D = 24
CH_ZONES = ("dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3")
YUNLIN_UNITS = ("dt.tercile.west", "dt.tercile.mid", "dt.tercile.east")

#: Seasonal amplitude (plan S4 step 8): per pixel, least squares over the pre-origin run P
#: epochs of a + b t + annual + semiannual; amplitude of the annual term (mm, vertical).
SEAS_WINDOW = ("2020-01-01", "2022-12-31")
SEAS_MIN_EPOCHS = 60
DENSITY_SCALES_KM = (2.0, 5.0, 10.0)
DENSITY_PRIMARY_KM = 5.0
PLACEBO_RING_KM = (15.0, 20.0)
CENSUS_YEARS = (2020, 2022)
BOOT_SEED = 4242
BOOT_N = 1000
BOOT_BLOCK_KM = 5.0
BATTERY_MIN_SPLITS = 6           # of 8

#: Ascending/descending decomposition (2025, after the fact).
DECOMP_SPAN = ("2025-01-23", "2025-12-11")

LICENCE = ("Copernicus Sentinel-1 derived-work terms: 'Contains modified Copernicus Sentinel "
           "data [2020-2026] analysed by the COMET LiCSAR system'; LiCSAR (Lazecky et al. "
           "2020, Remote Sensing 12(15) 2430) and LiCSBAS (Morishita et al. 2020, Remote "
           "Sensing 12(3) 424) must be cited.")

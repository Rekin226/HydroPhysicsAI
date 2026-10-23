"""WP H3 configuration (plan WP H3, 2.4-2.5, 4.4-4.5).

Every threshold here comes from the plan text (WP H3: >= 80 % observed monthly coverage,
layer codes 1-4, on the grid, no extrapolation beyond a record end), from H2's tested
envelope (``h2_heads_fill.config``: >= 36 observed months, internal runs <= 24 months, leading
edges <= 66 months) or from A1's harness (seeds, coverage rule gate-3). Nothing here was
chosen on data dated 2023-01 or later (U7), and no threshold was set after looking at a
coverage or error result of the candidate wells.

No path in this file names a user or a scratchpad: the work and evidence directories live in
the gitignored ``results/dataprep/reports/H3_evidence/`` (override with ``H3_EVIDENCE``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
WP = "H3"

# ----------------------------------------------------------------------------- inputs
L1_DIR = REPO / "results/dataprep/heads/L1_clean/v3"            # H1a product for use
H2_PRE_DIR = REPO / "results/dataprep/heads/L2_filled_preorigin/v4"   # H2 current pre-origin
UNITS_DIR = REPO / "results/dataprep/validation_units/v3"       # the 158 model wells
GRID_NPZ = REPO / "results/twin_forward/datum_gate_innov.npz"    # model grid (task)
H2_EVIDENCE = REPO / "results/dataprep/reports/H2_evidence"
H2_PARAMS = H2_EVIDENCE / "design" / "design_params.json"       # frozen bridge sds (H2 v1)
#: L0, read-only: the twin's own cache (for the twin's QC rule) and its station table.
AMP_WELLS = REPO / "AMP_V2/data/wells"
AMP_STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"

# ----------------------------------------------------------------------------- outputs
PRODUCT = "heads/recovered_wells"
EVIDENCE = Path(os.environ.get("H3_EVIDENCE", REPO / "results/dataprep/reports/H3_evidence"))
RUN_DIR = EVIDENCE / "run_tables"
H2_WORKDIR = EVIDENCE / "h2_workdir"         # H2 runner cache (base matrix) for H3 runs only

# ----------------------------------------------------------------------------- calendar
START = "2012-01"
PRE_END = "2022-12"                  # forecast origin (plan 4.2)
N_MONTHS_PRE = 132

# ----------------------------------------------------------------------------- inclusion
#: The settled coverage definition (plan H3 'Reconcile counts'): the share of the 132
#: pre-origin months that are L1 v3 C20 months (>= 20 clean days) with flag 0/1, i.e. the
#: observed months H2 and the U2 sigma_obs rule use. Fills never count.
COVERAGE_DEFINITION = "l1_c20"
MIN_COVERAGE = 0.80                  # plan H3: >= 80 % monthly coverage on observed months
COUNT_THRESHOLD = 0.30               # the brief's 'at least 30 % coverage' reconciliation
LAYERS = ("1", "2", "3", "4")
#: H2's tested envelope, applied to observed months only (h2_heads_fill.config).
MIN_OBS_MONTHS = 36
MAX_INTERNAL_RUN = 24
MAX_LEAD = 66
#: The record end is the last pre-origin month with an L1 flag-0/1 day. Months after it are
#: flag 6 (beyond the record end; plan 2.4, H3) and never enter calibration. A candidate's
#: record must reach the origin month: a trailing run longer than this disqualifies it
#: (plan: 'none with a trailing gap'; the records that stop on 2018-12-31 enter only if later
#: records are found).
MAX_TRAIL = 0

# ----------------------------------------------------------------------------- registered test
PREREG_ID = "H3.recovery.v1"
TEST_D = "H3.d_wells"
TEST_SNAP = "H3.snapshot"
#: Family d of H2 (one random daily gap of 5-60 days per eligible well per realization,
#: seed 1000*ord('d') + r): realizations never generated before this WP. H2 used r 0-4
#: (design), 10-19 (H2.fresh.d) and 60-69 (H2.fresh3.d). 60 realizations: a dry run on the
#: design realization r = 0 (counts only) kept a scored window at 10 of 16 test wells, so
#: about 60 draws are needed for k >= 20 windows per well.
FRESH_D = range(100, 160)
MIN_WINDOWS_PER_WELL = 20            # A1 gate-3 envelope: k >= 20 clusters

#: twin rule (hydrophysics.twin.heads.build_head_field defaults)
TWIN_MIN_COV = 0.80
TWIN_MAX_GAP_DAYS = 180.0

"""Declare I1.X5R1.v2_postfailure, the ONE exploratory follow-up I1.X5R1.v1 allows, kind
'declaration', before any of its columns is fitted.

Question: I1.X5R1.v1 failed, and q0c (Q0 + the Changhua census) failed the same leveling and
Changhua county-mean gates, so the failure belongs to adding Changhua pumping. With it the
leveling fit of the three-zone column collapses IN SAMPLE (0.66 -> 0.34; Changhua 0.08 ->
-0.67 out of fold; Yunlin mid zone 0.74 -> 0.33). Is that the column sharing one parameter set
per zone across the two counties, or are the Changhua-pumping heads themselves inconsistent
with leveling?

Design: no flow refit. Each arm's theta is copied with every zonal flow parameter duplicated
into its Changhua copy and all copies tied (``zone_county_tied``), so the flow model and its
heads are unchanged (checked: max |dh| over the hindcast < 1e-6 m before any column is fitted)
while calibrate_coupled fits the six-zone county-split column. Same column recipe and leveling
folds (seed 2027) as batch 27; same forward run.

Usage (repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_energy_volume/prereg_x5r1_followup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
REP = "results/dataprep/reports/I1_X5R1/report.json"


def doc() -> dict:
    base = json.loads((REPO / "results/dataprep/preregistration/I1.B.v1.json").read_text())
    d = {k: base[k] for k in ("burn_in", "folds", "gate_scores", "zones")}
    d.update({
        "id": "I1.X5R1.v2_postfailure", "wp": "I1", "kind": "declaration",
        "followup_of": "I1.X5R1.v1",
        "title": "Post-failure exploratory follow-up of rung X5R1: does a county-split "
                 "(six-zone) column, refitted on unchanged heads, recover the leveling fit "
                 "lost when Changhua pumping is added?",
        "priors": (
            f"Seen before writing ({REP}, sha256 fc61df22bb6d...): leveling out-of-fold R2 Q0 "
            "0.653, q0c 0.335, x5r1 0.344; by county Changhua 0.080 / -0.665 / -0.684, rest "
            "0.695 / 0.395 / 0.411; in-sample 0.659 / 0.343 / 0.353; Changhua head amplitude "
            "0.36 / 1.16 / 1.19. Known risk (rung R1c, 2026-10-23): a six-zone column made the "
            "forward run uplift after 2022."),
        "arms": {
            "q0_cs": "Q0 heads, six-zone column: the split's own gain (control)",
            "q0c_cs": "q0c heads (Q0 + Changhua census), six-zone column: the question",
            "x5r1_cs": "x5r1 heads (doubled irrigation volume), six-zone column"},
        "recipe": (
            "theta: results/twin_runs/stage3_datum_gate_<arm>/stage3_theta.json copied to "
            "stage3_datum_gate_<arm>_cs with every zonal key duplicated as <key>_changhua, "
            "meta zone_county_split true, the default county map record, zone_county_tied all "
            "four zones; heads-identity check (max |dh| < 1e-6 m) before any fit. Column: "
            "calibrate_coupled --target leveling --configs zonal --hpc0-guard-days 365 --seed "
            "2027 --device cuda. Forward: batch 27's _forward. Code: main 5f3af18, worktree "
            "~/HydroPhysicsAI-x5run (unchanged)."),
        "primary_model": "q0c_cs, read against q0_cs - Q0 (the split's gain on Q0 heads)",
        "inputs": ["results/dataprep/preregistration/I1.X5R1.v1.json", REP],
        "tests": NA("exploratory post-failure follow-up on the design-time partition; never "
                    "validation; cannot promote"),
        "fresh_units": NA("exploratory"),
        "thresholds": {
            "reading_fixed_now": [
                "SHARED COLUMN supported if q0c_cs leveling out-of-fold R2 >= 0.633 (Q0 - "
                "0.02) AND its Changhua R2 >= 0.080 (Q0) AND (q0c_cs - q0c) exceeds (q0_cs - "
                "Q0) by >= 0.10",
                "HEADS INCONSISTENT if q0c_cs leveling R2 < 0.50: the split column cannot "
                "turn the Changhua-pumping heads into the observed subsidence; the next "
                "suspect is the heads (trend or storage), not the column",
                "PARTIAL otherwise",
                "the same reading for x5r1_cs is reported beside it"],
            "reported_not_gating": [
                "leveling R2 in sample, out of fold, by county and by zone; column parameters",
                "forward 2023-25 rate by county and fan subsidence 2023-32; any county with a "
                "negative 2023-25 rate flags the R1c six-zone uplift artefact"]},
        "attempt_policy": {
            "scored_runs": "one column per arm; no refit after any metric is seen",
            "engineering_retries": "free before any metric is written (crash, OOM, failed "
                                   "heads-identity check)",
            "pre_declared_alternatives": [], "exploratory_followups_max": 0,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "interpretation": (
            "A supported SHARED COLUMN says the leveling collapse is a column-capacity defect "
            "and Changhua pumping can enter the model once the column is split by county; it "
            "would take a new registered rung (with the R1c forward artefact checked) to use "
            "it. HEADS INCONSISTENT says the extra Changhua pumping moves heads in a way the "
            "subsidence record does not show, so its magnitude or placement is wrong for "
            "this model. Neither outcome changes I1.X5R1.v1's FAIL."),
        "no_prior_look_statement": "No county-split column exists for q0, q0c or x5r1 heads; "
                                   "only the I1.X5R1.v1 report numbers above have been seen.",
        "written_by": "dataprep_staging/x5_energy_volume/prereg_x5r1_followup.py",
    })
    return d


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    print(*prereg3.write_prereg(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Declare the X5.R1 twin rung (stage 2 of route 1: the energy-to-volume fix), kind
'declaration', before batch 27 runs. The gates are I1.B.v1's, copied verbatim, against Q0.

Arms (batch 27, results/twin_runs/par27):

* ``q0c``: Q0 + the four-county census (model_census/v1, the twin's own dedupe-cap cleaning)
  at the current conversion -- rung B's B1 without the public supply. Reported only: it
  separates "Changhua pumping added" from "irrigation volume doubled";
* ``x5r1`` (primary): Q0 + the frozen kept census with irrigation-class kWh x m(county),
  Changhua 2.1085 and Yunlin 2.2220 (WRA irrigation groundwater per kept kWh, Jan-May
  2011-22, over the twin's 3.937 m3/kWh), ``--meter-filter none``;
* ``x5r1t`` (pre-declared alternative): the same with a township-varying factor shrunk to
  the county mean. Reported beside the primary, never substituted.

Usage (from the repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_energy_volume/prereg_x5r1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
S2 = "results/dataprep/x5_energy_volume/stage2_inputs"


def doc() -> dict:
    base = json.loads((REPO / "results/dataprep/preregistration/I1.B.v1.json").read_text())
    gates = base["thresholds"]["passes_against_Q0_if_all"][:6]
    d = {k: base[k] for k in ("burn_in", "folds", "gate_scores", "zones")}
    d.update({
        "id": "I1.X5R1.v1", "wp": "I1", "kind": "declaration",
        "title": "Rung X5.R1: the energy-to-volume fix. Irrigation-class census kWh x the WRA "
                 "conversion over the twin's (Changhua 2.11, Yunlin 2.22), four-county census; "
                 "township-varying factor as the declared alternative",
        "priors": (
            "Inferred-input rung (user direction 2026-10-24/25: infer missing quantities from held "
            "data; inferred values enter only as model inputs, never as observations). Route 1 "
            "(X5.energy_volume.*): the twin converts kept census kWh at 3.937 m3/kWh (eta 0.5, "
            "median lift 6.65 m, h_extra 40 m); WRA irrigation groundwater per kept irrigation kWh "
            "(Jan-May 2011-22, U7-clean) is 8.30 (Changhua) and 8.75 (Yunlin); the energy dropped "
            "by the cleaning does not look like irrigation pumping (STEP 0), so the gap is read as "
            "conversion. The independent head route (c) passed its registered test on the median "
            "(0.80; 0.68 without the one well looked at before registration, disclosed) but its "
            "interval 0.31-1.90 also covers the current conversion: weak corroboration. "
            f"Recipe = Q0 (results/twin_runs/par20/jobs.sh RECIPE, verbatim) + --pump-census/"
            f"--pump-kwh: q0c results/dataprep/pump_electricity/model_census/v1 (default cleaning); "
            f"x5r1 {S2}/county with --meter-filter none; x5r1t {S2}/township with --meter-filter "
            "none (built by dataprep_staging/x5_energy_volume/build_stage2_inputs.py). Precheck "
            "(results/dataprep/x5_energy_volume/stage2_precheck.json): the frozen m1 inputs "
            "reproduce the twin's own cleaned model_census/v1 energy per class, cell and month "
            "exactly; the county arm raises irrigation energy 2303 -> 4988 GWh (2012-22) and leaves "
            "every other class unchanged. 3-zone leveling column. Code: main 5f3af18, clean "
            "worktree ~/HydroPhysicsAI-x5run (par27/py.sh)."),
        "primary_model": "x5r1 (stage3_datum_gate_x5r1); x5r1t is the pre-declared alternative, "
                         "reported beside it, never substituted; q0c is reported only",
        "inputs": ["results/dataprep/preregistration/I1.B.v1.json",
                   "results/dataprep/reports/X5.energy_volume.c_vs_a.v1/report.json",
                   "results/dataprep/reports/X5.energy_volume.c_vs_a.v1/disclosure_2026-10-25.json",
                   "results/dataprep/reports/X5.energy_volume.step0_ab.v1/report.json",
                   "results/dataprep/x5_energy_volume/twin_rung_spec.json",
                   "results/dataprep/x5_energy_volume/stage2_precheck.json",
                   f"{S2}/county/build_info.json", f"{S2}/township/build_info.json",
                   f"{S2}/county/kwh.parquet", f"{S2}/township/kwh.parquet"],
        "tests": NA("design-time partition seed 2027; promotion needs a confirmation on the "
                    "never-scored partition seed 2029"),
        "fresh_units": NA("see the confirmation rule"),
        "thresholds": {
            "passes_against_Q0_if_all": gates,
            "promotion": ["only if every gate passes for x5r1 (or, if x5r1 fails, for x5r1t as "
                          "the pre-declared alternative): rerun Q0 and the passing arm on the "
                          "never-scored partition (calibrate_flow --seed 2029 --fold-site-tol-m "
                          "200, column --seed 2029) with the same gates against Q0"],
            "reported_not_gating": [
                "q0c on the same metrics (attribution: census added vs volume doubled)",
                "P2f Yunlin ratio (score_p2f_rung.py) for every arm",
                "fan and county irrigation volume (Mm3/yr) against WRA 2011-22",
                "calibrated T and S by zone against Q0 and q0c (A0: storage and volume trade "
                "off), and T against Liu et al.'s 58-6034 m2/day",
                "Changhua and Yunlin amplitude ratio and interannual correlation",
                "forward fan 2023-32 subsidence"]},
        "attempt_policy": {
            "scored_runs": "one per arm on seed 2027; no rerun after any gate metric is seen",
            "engineering_retries": "free before any metric is written (crash, OOM, NaN, missing "
                                   "file), each logged in gap_queue.log",
            "pre_declared_alternatives": ["x5r1t"],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "interpretation": (
            "Disclosed in advance: heads constrain Q/T, so doubling irrigation volume is likely "
            "to raise the calibrated T (or lower S) rather than move the head fit; a pass then "
            "says the corrected volume is compatible with the heads, not that heads confirm it, "
            "and the calibrated T is read against the literature range. If x5r1 passes, the "
            "corrected conversion becomes the volume-consistent base for later rungs (route 4's "
            "Changhua water term). If it fails on Changhua gates while q0c fails the same way, "
            "the failure belongs to adding Changhua pumping, not to the conversion; q0c is the "
            "comparison that decides which. A fail does not make the WRA conversion wrong: it "
            "says this model cannot absorb it with its current storage and transmissivity "
            "freedom."),
        "no_prior_look_statement": "No twin run with these inputs exists; no gate metric of "
                                   "any arm has been computed.",
        "written_by": "dataprep_staging/x5_energy_volume/prereg_x5r1.py",
    })
    return d


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    path, sha = prereg3.write_prereg(d)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

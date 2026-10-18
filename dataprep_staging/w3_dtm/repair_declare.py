"""Write the A1 declaration ``W3r.corrected_test_design.v1`` (harness.prereg3).

It records, before any residual of the repair round is computed:

* the diagnosis (snapshot of reports/W3.repair.md sections 1-4) and the corrected-test design
  (``repair_wells.DESIGN``, code hashed);
* the metadata-only dry run: only 3 wells at 2 fan sites carry a GroundHeight surveyed in
  2005-2022 (minimum 20 sites), and 0 fresh non-fan sites, so the corrected test is NOT
  evaluable and is not registered as a test or run;
* the leveling-only size of the survey-epoch ambiguity of the older GroundHeight values;
* the after-the-fact analysis that will follow (``repair_aftfact.py``): exactly which
  numbers are computed and that all of them are reported, never gating.

Usage: python -I repair_declare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common.paths import REPORTS_DIR, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402
from w3_dtm import repair_wells as rw  # noqa: E402

DECL_ID = "W3r.corrected_test_design.v1"
EVID = REPORTS_DIR / "W3_evidence" / "repair"
UNIT = "W3r:fan_wells_groundheight_consistent"


def main() -> int:
    dry = json.loads((EVID / "dryrun.json").read_text(encoding="utf-8"))
    amb = json.loads((EVID / "epoch_ambiguity.json").read_text(encoding="utf-8"))
    t = rw.eligibility(rw.station_table())
    import pandas as pd
    est = pd.to_numeric(t["EstablishDate"], errors="coerce")
    rules = ["E1_coords_dtm", "E2_groundheight", "E3_stickup", "E5_flat",
             "E6_no_postorigin_edit"]
    s = t[(t["set"] == "fan") & t[rules].all(axis=1)
          & (est <= int(rw.DESIGN["establish_window"][1]))]
    ids = [f"station_meta:{x}" for x in sorted(s["sid"])]
    na = prereg3.not_applicable
    doc = {
        "id": DECL_ID,
        "wp": "W3",
        "title": "W3 repair round: corrected test against surveyed ground at monitoring wells "
                 "(TWVD2001-era surveys) found not evaluable; after-the-fact analysis declared",
        "kind": "declaration",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP W3 (acceptance |median "
                        "bias| <= 0.3 m); repair rules a-e of the 2026-10-17 repair round",
        "tests": na("no test is run: the corrected test has 2 eligible fan sites (3 wells) "
                    "against its minimum of 20, and no fresh site"),
        "zones": na("station metadata; no zones"),
        "priors": na("no priors"),
        "burn_in": na("no model run"),
        "folds": na("no test is run; the after-the-fact analysis reports 10 km "
                    "block-bootstrap intervals (w3_dtm.validation.block_ids at the twin grid "
                    "origin, 2000 draws, seed 1000)"),
        "fresh_units": na("no fresh unit: the 12 WRA wells outside the fan set inside the DTM "
                          "coverage carry 2 GroundHeight values, both inconsistent with "
                          "WellElevation (rule E3)"),
        "external_units": {UNIT: {
            "description": "fan monitoring stations whose agency GroundHeight is a real value "
                           "consistent with WellElevation (rules E1, E2, E3, E5, E6 of "
                           "w3_dtm/repair_wells.py, any establishment date up to 2022-12-31); "
                           "members are station-metadata records, not head records; already "
                           "looked at by W3 diagnostics (2026-10-12, aggregate GroundHeight "
                           "minus DTM point) and H1b (2026-10-13, per-station agency minus "
                           "DTM)",
            "ids": ids}},
        "thresholds": {
            "original_gate_unchanged": "|median bias| <= 0.30 m (plan WP W3; registered "
                                       "W3.dtm_heldout_bias measured -0.842 m, failed)",
            "corrected_test_T1": "|median over sites| <= 0.30 m, evaluable with >= 20 sites; "
                                 "a pass would have been conditional (second look) and would "
                                 "not have opened the product gate by itself",
        },
        "gate_scores": na("declaration: no gate is scored"),
        "primary_model": {
            "corrected_test_design": rw.DESIGN,
            "code": "dataprep_staging/w3_dtm/repair_wells.py (hashed in inputs)",
            "residual": "r = H_pt + [u(EstablishDate) - u(t_sheet)] - GroundHeight; H_pt = "
                        "bilinear 20 m DTM of ground_elevation/v3 at the station; u = IDW "
                        "leveling displacement, main campaigns 2005-2021 with >= 400 "
                        "benchmarks",
            "eligibility_rules": ["E1 finite coordinates and DTM point",
                                  "E2 GroundHeight finite and not 0.0",
                                  "E3 |WellElevation - GroundHeight| <= 5 m when WellElevation "
                                  "is finite",
                                  "E4 EstablishDate 2005-01-01..2022-12-31",
                                  "E5 DTM sd in the 100 m window <= 2 m",
                                  "E6 ModificationDate <= 2022-12-31"],
        },
        "dry_run_metadata_only": {"eligibility_counts": dry["eligibility_counts"],
                                  "displacement_model": dry["displacement_model"],
                                  "gap_scaled_equals_w3_model_max_abs_m":
                                      dry["gap_scaled_equals_w3_model_max_abs_m"],
                                  "synthetic_check": dry["synthetic_check"],
                                  "eligible_units_T1": dry["eligible_units"]["fan"]},
        "epoch_ambiguity_leveling_only": {k: amb[k] for k in (
            "stations_passing_E1_E2_E3_E5_E6", "sites", "established_before_2003",
            "established_2005_2022", "epoch_correction_if_survey_at_establishment_m",
            "epoch_correction_if_survey_at_modification_m",
            "ambiguity_u_establish_minus_u_modification_m", "share_ambiguity_gt_0.30_m")},
        "decision": (
            "The corrected test is not evaluable and is neither registered as a test nor run. "
            "No reference in the data at hand is at once at bare ground, independent of the "
            "benchmark marks, of known survey epoch, in TWVD2001, and available at >= 20 "
            "sites; fresh units of that kind do not exist. The original gate stays FAILED "
            "(-0.842 m), no product method changes (no defect diagnosed), and no new "
            "ground_elevation version is built."),
        "after_the_fact_declared": {
            "script": "dataprep_staging/w3_dtm/repair_aftfact.py (hashed in inputs)",
            "units": UNIT,
            "numbers_all_reported": [
                "site and well medians, IQR, MAD-sd and 10 km block-bootstrap 90 % "
                "intervals of r under assumption A (survey at EstablishDate) and assumption "
                "B (survey at ModificationDate)",
                "the same without epoch correction",
                "the same with the containing 1 km cell mean (A and B)",
                "site medians by county prefix (07 Changhua, 09 Yunlin, other) under A and B",
                "the 3 wells surveyed 2014-2016 listed individually",
                "epoch-correction distributions under A and B"],
            "role": "after-the-fact analysis only: never gating, never fed back into "
                    "ground_elevation, never a threshold or selection; it shows the range "
                    "the well reference allows and cannot choose between A and B",
        },
        "looked_at_before": [
            "W3 report and evidence (2026-10-12): primary -0.842 m; point -0.384 m; site "
            "types; local windows; provenance; agency GroundHeight minus DTM point +0.415 m "
            "(n 182), WellElevation +0.686 m (n 328), BaseElevation +0.334 m (n 314)",
            "H1b report (2026-10-13): agency minus DTM at 2017.5 +0.416 / +0.439 m; stick-up "
            "proxy minus DTM +0.098 m at replaced stations; WellElevation minus DTM 0.712 vs "
            "0.654 m",
            "this session: station metadata counts, boolean DTM coverage of non-fan WRA and "
            "CWA stations (no DTM value printed), the dry run and the epoch-ambiguity numbers "
            "above (no GroundHeight or DTM value enters them); no agency-minus-DTM residual "
            "was computed in this round before this declaration"],
        "inputs": [str(p) for p in (
            EVID / "diagnosis_written_2026-10-17T1714Z.md", EVID / "dryrun.json",
            EVID / "epoch_ambiguity.json", HERE, HERE.parent / "repair_dryrun.py",
            HERE.parent / "repair_epoch_ambiguity.py", HERE.parent / "repair_aftfact.py",
            *rw.code_files(), *rw.input_files(),
            rw.REPO / "results/dataprep/preregistration/W3.dtm_heldout_bias.json")],
        "interpretation": (
            "Diagnosis (reports/W3.repair.md sections 1-4, snapshot hashed): the registered W3 "
            "test compares bare-earth ground with leveling-benchmark marks set on raised "
            "structures; about half of the -0.84 m is cell-scale siting (DTM only, predicted "
            "before the test) and most or all of the point residual (-0.38 m) is "
            "site-type-dependent siting; a datum offset cannot be separated with benchmarks. "
            "A corrected test needs a bare-ground reference of known epoch in TWVD2001. Agency "
            "GroundHeight is the only bare-ground reference in hand: 162 of its 165 usable "
            "values come from stations established 1968-2000 whose survey epoch is "
            "undocumented (establishment or the 2009-2011 bulk modification; leveling puts "
            "0.33 m of subsidence between the two in the median, 56 % above 0.30 m) and may "
            "predate TWVD2001. Only 3 wells at 2 sites qualify. The product therefore cannot "
            "pass honestly in this round; it stays excluded with the gate failed."),
    }
    path, sha = prereg3.write_prereg(doc)
    print(rel_to_repo(path), sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

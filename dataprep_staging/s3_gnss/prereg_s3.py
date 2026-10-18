"""WP S3: write the acceptance pre-registration with the A1 tool, BEFORE ``run_validation.py``.

    python -I dataprep_staging/s3_gnss/prereg_s3.py --tune TUNE_JSON

Registers ``S3.gnss_clean_acceptance.v1``: the injected-step test (A1) as a test on new
synthetic units, the 2021-extra test (A2) as a declared criterion (its pairs already carry an
unregistered look, recorded by ``record_looks.py``), and the reported, non-gating analyses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build as B  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg  # noqa: E402

SPEC = "results/dataprep/reports/S3_evidence/acceptance_spec.json"
UNITS = str(HERE.parents[1] / "results/dataprep/validation_units/v1/units.json")


def doc(tune: str) -> dict:
    t = json.loads(Path(tune).read_text())
    na = prereg.not_applicable
    code = [rel_to_repo(HERE / c) for c in B.CODE if (HERE / c).is_file()]
    return {
        "id": B.PREREG_ID, "wp": "S3", "kind": "test",
        "title": "gnss_clean acceptance: injected steps (A1, test) and the 2021 extra "
                 "compaction after the common-mode removal (A2, declared criterion)",
        "tests": [{
            "test_id": "S3.A1_injected_steps_preorigin",
            "data_scope": "gnss_injection:preorigin_background:seed303",
            "fresh_units": ["s3.injection.seed303.wra.preorigin",
                            "s3.injection.seed303.ngl.preorigin"],
            "external_units": {
                "s3.injection.seed303.wra.preorigin":
                    "synthetic: one injected step per WRA station per family run, numpy "
                    "default_rng(303), 4 sizes x 10 repetitions, on the WRA series cut at "
                    "2022-12-31 (not in validation_units/v1: synthetic injections)",
                "s3.injection.seed303.ngl.preorigin":
                    "synthetic: same, NGL family (17 stations) cut at 2022-12-31",
                "s3.injection.seed101.wra.preorigin": "design-time (tune r1/r2 stage 1)",
                "s3.injection.seed101.ngl.preorigin": "design-time (tune r1/r2 stage 1)",
                "s3.injection.seed202.wra.preorigin": "design-time (tune r1/r2 stage 2)",
                "s3.injection.seed202.ngl.preorigin": "design-time (tune r1/r2 stage 2)"},
            "design_time_units": ["s3.injection.seed101.wra.preorigin",
                                  "s3.injection.seed101.ngl.preorigin",
                                  "s3.injection.seed202.wra.preorigin",
                                  "s3.injection.seed202.ngl.preorigin"],
            "procedure": "Pre-origin variant: every WRA (2020-01..2022-12; LNJS and YSLL "
                         "dropped per plan, ANES processed but outside the common modes) and "
                         "NGL (record start..2022-12, 17 stations: the 16 of acceptance_spec.json plus "
                         "JSAN, fetched after the spec as the only NGL station inside an "
                         "active twin cell) series cut at 2022-12-31; "
                         "full product pipeline (3 passes, detection common mode, plan "
                         "catalogue and NGL metadata dated <= 2022-12). One injected step per "
                         "station per family run; sizes 2, 3, 5, 10 cm, random sign, 10 "
                         "repetitions, seed 303; injection days as injection.eligible_days "
                         "(observed, >= 90 d from the record ends, >= 60 d from any step, "
                         "excursion or untied segment of the un-injected run, >= 20 "
                         "observations on each side within 30 d). Recovered = a step within "
                         "+-3 d of the injection day with |size error| <= 5 mm.",
            "threshold": "recovery rate >= 0.90 over all injected sizes, separately for WRA "
                         "and for NGL; every size class reported"}],
        "declared_criteria": [{
            "id": "S3.A2_2021_extra_survives_cm",
            "status": "declared, not blind: the A2 statistic was computed once by a smoke "
                      "test (round-r1 hyperparameters) before this file (look recorded in "
                      "looks.jsonl by record_looks.py); operationalisation and threshold "
                      "unchanged from acceptance_spec.json (10:47 CST, before the look)",
            "operationalisation": "as acceptance_spec.json S3.A2, computed on the pre-origin "
                                  "variant; GNSS = cleaned value, flags 0-1, minus the "
                                  "adopted WRA common mode wra_hf, monthly medians",
            "threshold": "same sign at >= 75 % of evaluable pairs (reference extra > 3 mm) "
                         "and pooled amplitude ratio >= 0.5"}],
        "reported_not_gating": [
            "A1 on the full records (WRA to 2025-03 incl. CYES, NGL to 2026-09), seed 303 "
            "(after-the-fact: a gate may not use 2023+ observations, U7)",
            "A1 secondary: harness family G (harness.gaps.inject_gnss_steps, r = 0..9), both "
            "variants",
            "A2 on the full variant", "A3 excess-rate stability at co-located rings "
            "(2020-01..2021-12 and 2023-02..2025-03)",
            "common-mode scatter tables (none, ngl_offfan, wra_hf, wra_onfan_fullband) and "
            "the PCA first-mode uniformity check",
            "scoring of results/twin_forward/datum_gate_innov.npz (rheology 0, baseline) "
            "against GNSS rates 2023-01..2025-03 (WRA) and 2023-01..2024-04 (NGL), raw and "
            "common-mode removed, with the model's own common mode removed the same way"],
        "zones": na("data-cleaning product; no zones"),
        "priors": na("no priors"),
        "burn_in": na("no model burn-in"),
        "folds": na("no folds; synthetic injections"),
        "fresh_units": ["s3.injection.seed303.wra.preorigin",
                        "s3.injection.seed303.ngl.preorigin"],
        "thresholds": {"A1": ">= 0.90 per family (WRA, NGL), pre-origin variant",
                       "A2": "same sign >= 0.75 and pooled ratio >= 0.5, pre-origin variant",
                       "A3": "report only (plan)"},
        "gate_scores": na("conjunctive gate: A1 (WRA), A1 (NGL) and A2 must each pass; no "
                          "weighted score"),
        "primary_model": {"pipeline": "dataprep_staging/s3_gnss (model.py r2: the scan does "
                                      "not apply the robust-fit clip mask)",
                          "detector": t["chosen_params"], "reference_stations": t["reference"],
                          "tune_file": rel_to_repo(Path(tune))},
        "inputs": code + [rel_to_repo(Path(tune)), SPEC] + B.inputs("full"),
        "interpretation": (
            "The gate (both product variants) is A1 (WRA and NGL) and A2 on the pre-origin "
            "variant. Deviation from acceptance_spec.json: the spec ran A1 on the full "
            "records; plan U7 forbids 2023+ observations in a gate, so the full-record A1 is "
            "reported only. A1's synthetic units are new, but their noise background is the "
            "same pre-origin data on which the detector was tuned (seeds 101/202; rule: max "
            "pooled 2 cm recovery subject to <= 0.05 off-fan false detections per "
            "station-year, 20 configurations), so the pre-origin recovery is optimistic "
            "relative to the full-record one, which adds 2023-26 background never used in "
            "tuning. Design-time expectation, disclosed before the test: the chosen "
            "detector recovered 0.82 of 2 cm steps (WRA 0.78, NGL 0.89) and 0.82 of 3 cm "
            "steps on the tuning seeds, below the 0.90 threshold, which was fixed at 10:47 "
            "CST before the tuning and is kept. A1 failing in either family fails the "
            "gate; thresholds are not changed after results. A2 is declared, not blind "
            "(see declared_criteria). JSAN joined the NGL set after the tuning; it is not a "
            "reference candidate, so the other stations' detection is unchanged."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.tune)
    errs = prereg.validate_prereg(d, UNITS)
    if errs or a.dry_run:
        print(json.dumps(errs or {"ok": True, "n_inputs": len(d["inputs"])}, indent=1))
        return 1 if errs else 0
    path, sha = prereg.write_prereg(d, units_registry=UNITS)
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

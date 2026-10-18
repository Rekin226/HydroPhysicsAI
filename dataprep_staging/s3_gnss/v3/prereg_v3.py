"""WP S3 repair round 3 (S3r): write the v3 acceptance pre-registration with the A1 tool
(``harness.prereg3``, the current version of ``harness.prereg``; external units declared with
members) BEFORE ``v3/run_validation.py`` computes any A1 score on the fresh units.

    python -I dataprep_staging/s3_gnss/v3/prereg_v3.py --tune TUNE_JSON [--dry-run]

Registers ``S3.gnss_clean_acceptance.v3``: the injected-step test (A1) on new synthetic units
(seed 707) for the v3 pipeline, with the v1/v2 procedure and thresholds unchanged; the A2
recomputation on the v3 pre-origin tables, declared as a second look on spent units; and the
reported analyses (harness family G r = 20..29, full-record A1, A3, twin scoring).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # dataprep_staging
sys.path.insert(0, str(HERE))              # v3 modules first
import build as B  # noqa: E402
import units_v3 as U  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

#: v3 modules that define the registered test (hashed; must not change before scoring).
#: build.py (product writer), the look recorder and the diagnostics are not registered inputs.
TEST_CODE = ["io_gnss.py", "model.py", "pipeline.py", "products.py", "build_tables.py",
             "injection.py", "validate.py", "run_validation.py", "tune.py", "units_v3.py",
             "prereg_v3.py"]

SPEC = "results/dataprep/reports/S3_evidence/acceptance_spec.json"
EARLIER = ["results/dataprep/preregistration/S3.gnss_clean_acceptance.v1.json",
           "results/dataprep/preregistration/S3.gnss_clean_acceptance.v2.json",
           "results/dataprep/reports/S3_evidence/repair_2026-10-13/acceptance/"
           "validation_preorigin.json",
           # frozen copy of the repair report's Parts 1-2 (diagnosis and design), written
           # just before the registration; the live report keeps growing
           "results/dataprep/reports/S3_evidence/repair_2026-10-18/"
           "S3.repair.parts1-2.snapshot.md"]


def design_expectation(t: dict) -> str:
    c = t["stage1"]["chosen"]
    h = {(r["size_order"], r["local_half"]): r for r in t["stage2_grid"]}[
        (t["chosen_size_order"], t["chosen_local_half"])]
    return (f"design run 3: stage 1 (seed 525) chose scan window {int(c['win'])} d, z_min "
            f"{c['z_min']} "
            f"(empirical-null units), delta_min {1000 * c['delta_min']:.0f} mm: pooled 2+3 cm "
            f"recovery {c['recovered_2_3cm']:.3f} (WRA {c['recovered_wra']:.3f}, NGL "
            f"{c['recovered_ngl']:.3f}), off-fan false rate "
            f"{c['false_per_station_year_offfan']:.3f} per station-year; stage 2 (seed 626) "
            f"chose the step-size model of order {t['chosen_size_order']} on +-"
            f"{t['chosen_local_half']} d: pooled 2/3/5/10 cm recovery "
            f"{h['recovered']:.3f} (WRA {h['recovered_wra']:.3f}, NGL {h['recovered_ngl']:.3f};"
            f" by size {json.dumps({k: round(v, 3) for k, v in h['by_size'].items()})}); "
            f"found-step size error sd {h['size_err_mm_sd_found']:.2f} mm; wra_hf window "
            f"{t['cm_evidence']['hp_window_selected']} d "
            f"({t['cm_evidence']['hp_window_selected_by']})")


def doc(tune: str) -> dict:
    t = json.loads(Path(tune).read_text())
    na = prereg3.not_applicable
    code = [rel_to_repo(HERE / c) for c in TEST_CODE]
    ext = {
        U.FRESH_A1[0]: {
            "description": "synthetic: one injected step per WRA station per family run, "
                           f"numpy default_rng({U.SEED_TEST}), 4 sizes x 10 repetitions, on the "
                           "v3 pre-origin WRA series (every cache station with >= 60 days to "
                           "2022-12-31; the v2 station set)", "none": U.SYN},
        U.FRESH_A1[1]: {
            "description": "synthetic: same, v3 pre-origin NGL family (11 fan-box stations + "
                           "JSAN) cut at 2022-12-31 (the v2 station set)", "none": U.SYN}}
    for u in U.DESIGN_UNITS:
        ext[u] = {"description": "earlier synthetic injection draws (tune seeds 101/202, v3 "
                                 "design seeds 505/515/616/525/626, failed-test seeds 303/404 "
                                 "used in the S3r diagnosis)",
                  "none": U.SYN}
    return {
        "id": B.PREREG_ID, "wp": "S3", "kind": "test",
        "title": "gnss_clean v3 acceptance: injected steps (A1, test, seed 707) after the S3r "
                 "repair of the detector; A2 recomputed as a declared second look",
        "tests": [{
            "test_id": "S3.A1_injected_steps_preorigin_v3",
            "data_scope": f"gnss_injection:preorigin_background:seed{U.SEED_TEST}",
            "fresh_units": list(U.FRESH_A1),
            "external_units": ext,
            "design_time_units": sorted(U.DESIGN_UNITS),
            "procedure": "v3 pre-origin variant with the v2 station sets: WRA 2020-01..2022-12, "
                         "every cache station with >= 60 days; NGL fan-box stations and JSAN, "
                         "record start..2022-12; reference = the unchanged pre-origin rule; "
                         "full v3 product pipeline (3 passes, detection common mode, plan "
                         "catalogue and NGL metadata dated <= 2022-12, step re-test), tuned "
                         "parameters from the tune file below. One injected step per station "
                         "per family run; sizes 2, 3, 5, 10 cm, random sign, 10 repetitions, "
                         f"seed {U.SEED_TEST}; injection days as injection.eligible_days "
                         "(unchanged since v1). Recovered = a step within +-3 d of the "
                         "injection day with |size error| <= 5 mm.",
            "threshold": "recovery rate >= 0.90 over all injected sizes, separately for WRA "
                         "and for NGL; every size class reported"}],
        "zones": na("data-cleaning product; no zones"),
        "priors": na("no priors"),
        "burn_in": na("no model burn-in"),
        "folds": na("no folds; synthetic injections"),
        "fresh_units": list(U.FRESH_A1),
        "thresholds": {"A1": ">= 0.90 per family (WRA, NGL), pre-origin variant (unchanged)",
                       "A2": "same sign >= 0.75 and pooled ratio >= 0.5 for each of "
                             "ngl_offfan and wra_hf, pre-origin variant (unchanged; scored "
                             "as a second look on spent units, see interpretation)",
                       "A3": "report only (plan)"},
        "gate_scores": na("conjunctive gate: A1 (WRA), A1 (NGL) and A2 (both common modes) "
                          "must each pass; no weighted score"),
        "primary_model": {"pipeline": "dataprep_staging/s3_gnss/v3 (v3 code)",
                          "detector": t["chosen_params"], "reference_stations": t["reference"],
                          "reference_stations_full_variant": t["reference_full"],
                          "tune_file": rel_to_repo(Path(tune))},
        "inputs": code + [rel_to_repo(HERE.parent / c) for c in B.CODE_V1]
                  + [rel_to_repo(Path(tune)), SPEC] + EARLIER
                  + B.inputs("full") + ["hydrophysics/twin/mechanism.py"],
        "interpretation": (
            "Context: two registered tests of this design family failed with these thresholds: "
            "v1 (seed 303) A1 WRA 0.816, NGL 0.891, A2 same sign 0.571; v2 (seed 404) A1 WRA "
            "0.863, NGL 0.8625, A2 same sign 0.50. This is the THIRD registered test; a pass "
            "would follow two failures and must be reported as such. The v3 detector answers "
            "defects diagnosed before any re-test (results/dataprep/reports/S3.repair.md "
            "Part 1) or found on design units (Part 2): D1 a noise-limited step-size estimate "
            "(R1': a robust level difference on +-local_half days of the series minus the "
            "station's global model and the detection common mode, without the free local "
            "slope that doubled its variance; the first R1, a quadratic trend on a longer "
            "window, rested on a wrong premise and its design run was stopped); D2 a noise "
            "scale for z that is mis-specified (AR(1) from lag-1) and estimated once before "
            "detection, so unmodelled real steps deflate z (R2: empirical-null z re-estimated "
            "each iteration, and R2b: time-local over +-365 d because noise is not "
            "stationary, e.g. ERLN; the same calibration for the step se and own-series z); D3 "
            "a gap-unaware sharpness test (R3); D4 the loop stopping at a max-z candidate "
            "failing the size condition (R4); D5 the excursion test returning at modelled "
            "steps and reading curvature as a box (R5); and, from design-time evidence, R6 a "
            "5-nearest-station WRA detection common mode. Station sets, catalogue, metadata, "
            "product common modes, the NGL detection common mode, the transient rule, the A1 "
            "procedure and every threshold are unchanged; ERLN, the noisiest NGL station (22 "
            "of 40 v2 trials failed), stays in. Design: three runs, each with new seeds (run 1 "
            "stopped, run 2 exposed R2b's defect, run 3 final; rules declared in tune.py "
            "before run 2): "
            + design_expectation(t) + ". Limitation: the fresh units are new random draws "
            "(seed 707), but the background series are the same real pre-origin series that "
            "every earlier test, the diagnosis and the design used; no unused GNSS background "
            "or window exists. The A1 outcome is therefore expected to track the design "
            "result within sampling noise (about 0.01-0.02 for 1,000 / 480 trials). Stated "
            "expectation: across the design seeds of runs 2-3 the pooled recovery was WRA "
            "0.88-0.90 and NGL 0.83-0.90, so A1 is about as likely to fail as to pass in each "
            "family (WRA more likely to fail: 0.883 on seed 626), and a pass in both families "
            "is unlikely. A2 is expected to fail. A2: its 14 pair "
            "units were scored in v1 and v2 and examined in the S3r diagnosis (decisive "
            "GNSS-ring disagreements at KTES and TKJS exist in the raw vendor data; the "
            "outcome depends on step handling at GFES, TKJS and YCES); no fresh co-located "
            "reference exists. A2 is recomputed on the v3 pre-origin tables with the "
            "registered v2 definition and reported as a SECOND LOOK: a pass would be "
            "conditional and carries no evidential weight; a failure keeps the gate failed. "
            "No corrected A2 test is registered. Gate rule: A1 must pass in both families on "
            "the fresh units and A2 for both common modes; because A2 can only be scored as a "
            "second look, a numerical pass of both gives the status 'conditional (second "
            "look)' and the product manifests keep gate.passed = false until the user "
            "decides; any failure gives 'failed'."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.tune)
    errs = prereg3.validate_prereg(d)
    if errs or a.dry_run:
        print(json.dumps(errs or {"ok": True, "n_inputs": len(d["inputs"]),
                                  "interpretation": d["interpretation"]}, indent=1,
                         ensure_ascii=False))
        return 1 if errs else 0
    path, sha = prereg3.write_prereg(d)
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

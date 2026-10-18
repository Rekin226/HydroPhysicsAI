"""WP S3 repair round: write the v2 acceptance pre-registration with the A1 tool prescribed by
A1 repair round 2 (``harness.prereg3``; external units declared with members), BEFORE
``v2/run_validation.py``.

    python -I dataprep_staging/s3_gnss/v2/prereg_v2.py --tune TUNE_JSON [--dry-run]

Registers ``S3.gnss_clean_acceptance.v2``: the injected-step test (A1) on new synthetic units
(seed 404) for the v2 pipeline, the 2021-extra criterion (A2) as a declared criterion that
must hold for each common mode the product offers for removal, and the reported analyses.
Thresholds are those of v1 (acceptance_spec.json, 10:47 CST), unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # dataprep_staging
sys.path.insert(0, str(HERE))              # v2 modules first
import build as B  # noqa: E402
import units_v2 as U  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

#: v2 modules that define the registered test (hashed; must not change before scoring).
#: build.py (product writer) and the look recorders are not registered inputs.
TEST_CODE = ["io_gnss.py", "model.py", "pipeline.py", "products.py", "build_tables.py",
             "injection.py", "validate.py", "run_validation.py", "tune.py", "units_v2.py",
             "prereg_v2.py"]

SPEC = "results/dataprep/reports/S3_evidence/acceptance_spec.json"
V1_PREREG = "results/dataprep/preregistration/S3.gnss_clean_acceptance.v1.json"
V1_VALIDATION = "results/dataprep/reports/S3_evidence/acceptance/validation_preorigin.json"


def design_expectation(t: dict) -> str:
    c = t["chosen"]
    return (f"chosen detector delta_min {c['delta_min'] * 1000:.0f} mm, z_min {c['z_min']}, "
            f"local_half {t['chosen_local_half']} d: pooled 2 cm recovery "
            f"{c['recovered_2cm']:.3f} (WRA {c['recovered_2cm_wra']:.3f}, NGL "
            f"{c['recovered_2cm_ngl']:.3f}), 3 cm {c['recovered_3cm']:.3f}, off-fan false "
            f"rate {c['false_per_station_year_offfan']:.3f} per station-year (seeds 101/202, "
            f"r3); wra_hf window {t['cm_evidence']['hp_window_selected']} d "
            f"({t['cm_evidence']['hp_window_selected_by']})")


def doc(tune: str) -> dict:
    t = json.loads(Path(tune).read_text())
    na = prereg3.not_applicable
    code = [rel_to_repo(HERE / c) for c in TEST_CODE]
    seeds = [f"s3.injection.seed{s}.{f}.preorigin" for s in (101, 202) for f in ("wra", "ngl")]
    ext = {
        "s3.injection.seed404.wra.preorigin": {
            "description": "synthetic: one injected step per WRA station per family run, "
                           "numpy default_rng(404), 4 sizes x 10 repetitions, on the v2 "
                           "pre-origin WRA series (every cache station with >= 60 days to "
                           "2022-12-31, LNJS, YSLL and ANES included)",
            "none": U.SYN},
        "s3.injection.seed404.ngl.preorigin": {
            "description": "synthetic: same, v2 pre-origin NGL family (11 fan-box stations + "
                           "JSAN) cut at 2022-12-31",
            "none": U.SYN}}
    for u in seeds:
        ext[u] = {"description": "design-time synthetic injections (tune rounds r1/r2/r3, "
                                 "seeds 101/202)", "none": U.SYN}
    return {
        "id": B.PREREG_ID, "wp": "S3", "kind": "test",
        "title": "gnss_clean v2 acceptance: injected steps (A1, test, seed 404) and the 2021 "
                 "extra compaction after the removal of each offered common mode (A2, "
                 "declared criterion)",
        "tests": [{
            "test_id": "S3.A1_injected_steps_preorigin_v2",
            "data_scope": "gnss_injection:preorigin_background:seed404",
            "fresh_units": ["s3.injection.seed404.wra.preorigin",
                            "s3.injection.seed404.ngl.preorigin"],
            "external_units": ext,
            "design_time_units": sorted(seeds),
            "procedure": "v2 pre-origin variant: WRA 2020-01..2022-12, every cache station "
                         "with >= 60 days (LNJS, YSLL and ANES are ordinary members); NGL "
                         "fan-box stations and JSAN, record start..2022-12; reference = the "
                         "pre-origin rule (off the twin grid, >= 2 years, |rate| <= 5 mm/yr) "
                         "over the plan's height-based candidates inside the box; full v2 "
                         "product pipeline (3 passes, detection common mode, plan catalogue and "
                         "NGL metadata dated <= 2022-12, step re-test |size| >= delta_min), "
                         "tuned parameters from the tune file below. One injected step per "
                         "station per family run; sizes 2, 3, 5, 10 cm, random sign, 10 "
                         "repetitions, seed 404; injection days as injection.eligible_days. "
                         "Recovered = a step within +-3 d of the injection day with |size "
                         "error| <= 5 mm.",
            "threshold": "recovery rate >= 0.90 over all injected sizes, separately for WRA "
                         "and for NGL; every size class reported"}],
        "zones": na("data-cleaning product; no zones"),
        "priors": na("no priors"),
        "burn_in": na("no model burn-in"),
        "folds": na("no folds; synthetic injections"),
        "fresh_units": ["s3.injection.seed404.wra.preorigin",
                        "s3.injection.seed404.ngl.preorigin"],
        "thresholds": {"A1": ">= 0.90 per family (WRA, NGL), pre-origin variant",
                       "A2": "same sign >= 0.75 and pooled ratio >= 0.5 for each of "
                             "ngl_offfan and wra_hf, pre-origin variant",
                       "A3": "report only (plan)"},
        "gate_scores": na("conjunctive gate: A1 (WRA), A1 (NGL) and A2 (both common modes) "
                          "must each pass; no weighted score"),
        "primary_model": {"pipeline": "dataprep_staging/s3_gnss/v2 (v2 code)",
                          "detector": t["chosen_params"], "reference_stations": t["reference"],
                          "reference_stations_full_variant": t["reference_full"],
                          "tune_file": rel_to_repo(Path(tune))},
        "inputs": code + [rel_to_repo(HERE.parent / c) for c in B.CODE_V1]
                  + [rel_to_repo(Path(tune)), SPEC, V1_PREREG, V1_VALIDATION]
                  + B.inputs("full") + ["hydrophysics/twin/mechanism.py"],
        "interpretation": (
            "Context: the v1 gate (S3.gnss_clean_acceptance.v1, seed 303) failed: A1 WRA "
            "0.816, NGL 0.891; A2 same sign 0.571. This v2 design responds to the independent "
            "verifier's findings, not to the size of the v1 shortfall: (1) the pre-origin "
            "station sets no longer rest on post-origin evidence: v1 dropped LNJS and YSLL and "
            "kept ANES out of the common modes for reasons seen only in 2023-25, so v2 keeps "
            "all three as ordinary pre-origin members; LSB0 (named by the plan together with "
            "its record to 2026-09) and v1's regional continuity set CKSV/JUNA/TCMS/S01R "
            "(added for their records after 2024-04) are kept out of the pre-origin variant as "
            "a precaution; (2) detected steps are re-tested after their final local size "
            "(|size| >= delta_min; the defect is visible in pre-origin data, e.g. FRES "
            "2021-09-12 at 3.9 mm in gnss_clean/preorigin/v1); (3) the wra_hf window follows "
            "its declared rule (in v1 the rule selected 21 d but the product kept 31 d); (4) "
            "A2 must hold for each offered common mode. The plan's LNJS/YSLL drops and the "
            "ANES flag (jump-cluster days flag 9) apply only to the full, after-the-fact "
            "variant. This is the second registered test of this design family, with the "
            "same thresholds; a pass here would follow one failed attempt and must be "
            "reported as such. Design-time expectation, disclosed before the test: "
            + design_expectation(t) + ". The 2 cm class therefore sits below 0.90 on the "
            "design seeds, and A1 is expected to fail again unless the larger sizes "
            "compensate. A2 is expected to fail: the same-sign failure in v1 came from KTES "
            "and TKJS disagreeing with their rings even without a common mode, and "
            "ngl_offfan removal gave a pooled ratio of 0.349 on gnss_clean/preorigin/v1. A1 "
            "failing in either family or A2 failing for either common mode fails the gate; "
            "thresholds are not changed after results."),
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

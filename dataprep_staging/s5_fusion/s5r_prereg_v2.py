"""WP S5 repair round: the declaration ``S5.subsidence_fused_rules.v2`` (rules of
subsidence_fused/v2), harness.prereg3; written BEFORE v2 is built.

    python -I s5r_prereg_v2.py [--dry-run]

Every amendment is decided AFTER v1's results and the verifier's findings were seen and is
labelled so; the registered gate tests T1 and T2 of S5.subsidence_fusion.v1 are NOT re-run,
re-scored or changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5r_const as R  # noqa: E402
import s5r_vconst as V  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

V1_PREREG_SHA = "6b56e526c4c334ac0b58648da93c64be0b409ab838963c60ade35a860c579a29"
ROUND1_CODE = ["s5const.py", "s5model.py", "s5data.py", "s5fit.py", "s5hyper.py"]
REPAIR_CODE = ["s5r_const.py", "s5r_vconst.py", "s5r_core.py", "s5r_stage2.py",
               "s5r_mtest.py", "s5r_prereg_m1.py", "s5r_prereg_v2.py", "s5r_product.py",
               "tests/test_s5r_stage2.py", "tests/test_s5r_core.py"]
SHARED_CODE = ["dataprep_staging/s1_rings/s1lib.py", "dataprep_staging/s1_rings/s1const.py",
               "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
               "dataprep_staging/common/hashing.py", "dataprep_staging/common/paths.py",
               "dataprep_staging/common/uses.py", "dataprep_staging/harness/coverage3.py",
               "dataprep_staging/harness/prereg3.py"]


def code_paths(names) -> list[str]:
    return [rel_to_repo(HERE / n) for n in names]


def pre_data_inputs() -> list[str]:
    from s1_rings import s1const as K1
    files = [C.LEVELING_PANEL, C.RING_STATIONS, C.TWIN_GRID,
             C.GNSS_PRE / "MANIFEST.json", C.GNSS_PRE / "stations.csv",
             C.GNSS_PRE / "wra" / "monthly.parquet", C.GNSS_PRE / "ngl" / "monthly.parquet",
             C.GNSS_FULL / "MANIFEST.json", C.GNSS_FULL / "stations.csv",
             C.PILES / "MANIFEST.json", C.PILES / "printed_monthly.parquet",
             C.WORK / "insar_runP_cells.npz", C.WORK / "insar_runP_cells.json",
             C.EVIDENCE / "hyper_pre.json"]
    files += [C.RING_DIR / f"{s}.parquet" for s in K1.SITES]
    return [rel_to_repo(f) for f in files]


def full_data_inputs() -> list[str]:
    files = [C.GNSS_FULL / "wra" / "monthly.parquet", C.GNSS_FULL / "ngl" / "monthly.parquet",
             C.GNSS_FULL / "wra" / "ncku_monthly.parquet"]
    files += [C.GNSS_FULL / v for v in V.GNSS_DAILY.values()]
    files += [C.INSAR_V2 / "MANIFEST.json", C.INSAR_V2 / "vertical_cells.npz",
              C.RINGS_BRIDGED / "MANIFEST.json", C.RINGS_BRIDGED / "ring_corrections.csv",
              C.UNITS_V3, C.EVIDENCE / "hyper_t1.json",
              C.DATAPREP / "preregistration" / f"{C.PREREG_ID}.json",
              C.DATAPREP / "subsidence_fused" / "v1" / "MANIFEST.json",
              C.DATAPREP / "subsidence_fused" / "v1" / "fused_annual.npz",
              R.EVIDENCE_R / "hyper_stage2.json"]
    return [rel_to_repo(f) for f in files]


def hashes(paths: list[str]) -> dict:
    from common.paths import from_repo
    return {p: sha256_file(from_repo(p)) for p in paths}


def m1_station_ids() -> list[str]:
    g = D.grid()
    gpre, _ = D.gnss(g, "pre")
    w = gpre[gpre.family == R.M1_FAMILY]
    return sorted({f"gnss:{R.M1_FAMILY}:{s}" for s in w.station.unique()})


# ============================================================================ declaration v2
AMENDMENTS = {
    "1_gnss_offset_segments": (
        "GNSS operator: every station series is split at each gap longer than 30 days between "
        "consecutive usable days (daily flag 0, 1 or 3) of its gnss_clean daily record "
        "(full/v2: wra/daily, ngl/daily, wra/ncku_daily); each segment gets its own diffuse "
        "offset (names goff:<family>:<station>:<k>, prior sd 1e4 mm); the station excess "
        "rate, family frame rate and family-month common modes are unchanged; a monthly value "
        "belongs to the segment of its days. 30 d is S3's tie_max_gap: S3's step scan (half-"
        "window 30 d) cannot test a step across a longer gap. POST-RESULT: decided after the "
        "verifier quantified SJES's uncorrected ~100 mm offset inside its 2022-12-22.."
        "2023-02-24 gap (plan S3 catalogue: 'unusable across the gap'; S1b) in v1 (+38/+37 mm "
        "mean 2022/2023 increments within 3 km). The rule is general, not SJES-specific; it "
        "also cuts the other gaps S1b listed (TKJS, STES, LYES, MDES, TCBS, XPES, JJES) and "
        "the 2020 GFES/TKJS gaps, so GNSS level continuity is no longer assumed across any "
        "untested gap."),
    "2_stepped_benchmarks": (
        "Leveling: after a first full-data fit (amendment 1 operator, leveling screening flags "
        "of hyper_pre.json), every benchmark with >= 4 surveys is scanned for one step on its "
        "residuals y - posterior-mean field, its own diffuse offset and random velocity (sd "
        "sig_rb) eliminated as in the operator (s5model.site_local): for each split with >= 2 "
        "surveys on each side, delta = s'W r / s'W s, sd = (s'W s)^-1/2; the largest |z| is "
        "flagged when |z| > 5 and |delta| >= 30 mm. A flagged benchmark gets a second diffuse "
        "offset from the step survey on (plan S5: 'stepped benchmarks flagged'); the list is "
        "published as leveling_steps.csv. The GNSS and ring/pile |z| > 5 screening pass is "
        "taken on the same first fit (as v1). POST-RESULT: decided after the verifier's T2 "
        "finding (8 increments >= 1.5 yr with 149 mm RMSE); T1 and T2 are not re-run."),
    "3_mask_excludes_flag6": (
        "mask_year = (vr_year >= 0.5) AND flag != 6: the extrapolated year 2026 is never in "
        "the VR mask (v1 marked 97 % of cells). mask_test (2023-01-01 -> 2026-01-01) is "
        "unchanged."),
    "4_monthly_field": (
        "Monthly field 2020-01..2025-12 (fused_monthly.npz): stage 2 of S5.monthly_field.v1 "
        "(s5r_stage2.py) on the InSAR run-F month-group residuals of the final full-data v2 "
        "fit, with the pre-origin stage-2 hyperparameters (hyper_stage2.json, fit_window_end "
        "2022-12). Value at month m (mid-month time t_m) = annual trend incl. seasonal pair "
        "and mean velocity at t_m + the posterior-mean anomaly of month m; sd^2 = annual "
        "posterior variance at t_m + stage-2 posterior variance (two-stage, independent). "
        "Delivered (flag 3) ONLY IF the registered test M1 of S5.monthly_field.v1 passed (as "
        "fixed in that registration), and then only where the cell lies in the month's InSAR "
        "footprint (a 4 km unit with a value that month, >= 30 cells) AND the plane-free "
        "anomaly VR >= 0.5 (support_vr); elsewhere value NaN, flag 8. The annual trend at "
        "mid-month (incl. the constant seasonal pair), the anomaly with its sd, support_vr and "
        "the footprint stay available as separate arrays for diagnosis. The planar part of "
        "the anomaly is not resolved by InSAR (plane per month projected out) and stays at "
        "its posterior (prior-driven) value inside the sd. M1's verdict and sd label go into "
        "the product card; the monthly sd is 'uncalibrated' unless M1's coverage passes."),
    "5_fit_window_end": (
        "fit_window_end = 2026-09, the latest fit_window_end of the inputs (gnss_clean/full/v2; "
        "plan 2.3 'inputs included'); the last observation month used by the fit (2025-12) is "
        "recorded as parameters.data_end."),
    "6_provenance": (
        "the manifest hashes every module the outputs depend on (round-1 modules, the repair "
        "modules, s1_rings s1lib/s1const, common guard/manifest/hashing/paths/uses/flags, "
        "harness coverage3/prereg3, s4_insar s4b_lib/s4b_evaluate/s4b_config that built the "
        "run-P cells) and every input read (incl. the GNSS daily tables, validation_units/v3, "
        "the run-P cells and their provenance JSON, both registrations and hyper_stage2.json)."),
    "7_labels_and_card": (
        "Every sd is labelled 'uncalibrated' in the product files (T1 coverage 0.631, rms z "
        "2.47; T2 0.940 over-covers). PRODUCT_CARD.md adds: the SJES artefact of v1 and its "
        "v2 treatment with the measured change, the T2 no-buffer deviation from plan 4.3, the "
        "non-converged optimiser stages, T1's optimism (stages B-D and screening from the "
        "main fit), R2's after-the-fact contamination (run F spans the origin; screening on the "
        "full fit), and that W3 cannot take an epoch correction from this product."),
    "8_sensitivity": (
        "sensitivity.json: v2 minus (a) the v1 operator (no GNSS segments, no leveling steps), "
        "(b) amendment 1 with SJES's breaks only, (c) rings and piles left out; annual "
        "increments 2022-2025 and the 2023-2026 change, RMS over cells, at SJES +-3/10 km, "
        "cells moved > 5 / > 10 mm. Report only."),
}

UNCHANGED = (
    "Everything else of S5.subsidence_fusion.v1 (sha256 " + V1_PREREG_SHA + ") stands: the "
    "annual model and every other operator, the lattice, knots, the pre-origin "
    "hyperparameters and depth fractions (hyper_pre.json, procedure H, not re-run), the VR "
    "rule (>= 0.5 against a leveling-only fit with the same hyperparameters and operators), "
    "the allowed uses (after_the_fact_analysis only: GNSS, InSAR and ring-interval inputs "
    "failed their gates) and the product gate: T1 and T2 as registered and scored on v1 "
    "(both FAILED). v2's operator changes are NOT covered by a registered gate test; the "
    "gate stays failed (gate.passed false) and is not re-tested on the design-exposed T1/T2 "
    "windows.")


def declaration_doc() -> dict:
    na = prereg3.not_applicable
    code = code_paths(ROUND1_CODE + REPAIR_CODE + ["s5product.py"]) + SHARED_CODE + [
        "dataprep_staging/s4_insar/s4b_lib.py", "dataprep_staging/s4_insar/s4b_evaluate.py",
        "dataprep_staging/s4_insar/s4b_config.py", "dataprep_staging/common/flags.py"]
    return {
        "id": R.DECL_ID, "wp": C.WP,
        "title": ("S5 repair round: rules of subsidence_fused/v2 (GNSS offset segments at "
                  "untested gaps, stepped benchmarks, flag-6 mask, monthly field, provenance, "
                  "labels); declared before v2 is built"),
        "kind": "declaration",
        "tests": na("no fresh test for the v2 rules: the product gate stays T1/T2 of "
                    "S5.subsidence_fusion.v1 (failed); the monthly component is tested by "
                    "S5.monthly_field.v1"),
        "zones": na("no zones"),
        "priors": {"gnss_segment_offsets": "diffuse (sd 1e4 mm)",
                   "leveling_step_offsets": "diffuse (restricted likelihood)",
                   "stage2": "hyper_stage2.json of S5.monthly_field.v1"},
        "burn_in": na("no spin-up"),
        "folds": na("no folds"),
        "fresh_units": na("none"),
        "thresholds": {"gnss_gap_days": V.GNSS_GAP_DAYS, "step_z": V.STEP_Z,
                       "step_min_mm": V.STEP_MIN_MM, "step_min_side": V.STEP_MIN_SIDE,
                       "vr_min": C.VR_MIN, "vr_a_min": R.VR_A_MIN,
                       "gate": "unchanged: T1 and T2 of " + C.PREREG_ID},
        "gate_scores": na("conjunctive gate of " + C.PREREG_ID + ", unchanged (failed)"),
        "primary_model": {"base": C.PREREG_ID, "base_sha256": V1_PREREG_SHA,
                          "amendments": AMENDMENTS, "unchanged": UNCHANGED,
                          "monthly_test": R.MTEST_ID},
        "inputs": code + pre_data_inputs() + full_data_inputs() + [
            rel_to_repo(C.DATAPREP / "preregistration" / f"{R.MTEST_ID}.json")],
        "interpretation": (
            "subsidence_fused/v2 is v1 with GNSS level continuity cut at every untested gap "
            "(SJES's ~10 cm in-gap offset no longer enters), stepped benchmarks given a second "
            "offset, the extrapolated year out of the VR mask, an InSAR-supported monthly "
            "field, complete provenance and corrected labels. Its gate stays failed and its "
            "allowed use stays after_the_fact_analysis. U7: amendment 1 was motivated by 2023+ "
            "GNSS evidence (SJES's post-gap level, seen by S1b and the verifier), amendment 2 "
            "by the design-exposed T2 rows; both are general rules for an after-the-fact "
            "product that selects no model and allows no pre-origin use, flagged here for the "
            "same U7 reading S1a asked the user to decide."),
        "anti_leakage": ("hyperparameters, depth fractions and stage-2 hyperparameters are "
                         "pre-origin (fit_window_end 2022-12); no 2023+ observation enters a "
                         "gate or threshold; the product forbids every pre-origin use."),
        "design_time_exposure": [
            "subsidence_fused/v1 (all files) and the verifier's findings were seen before this "
            "declaration",
            "GNSS daily-record gaps (dates and flags) were listed before this declaration "
            "(look 2026-10-17T03:23Z)",
            "the v1 T2 target table (8 long increments with 149 mm RMSE) was seen",
            "the pre-origin stage-2 hyperparameters (S5.monthly_field.v1 procedure, "
            "hyper_stage2.json) were seen; M1 was not yet scored when this was declared",
            "the product code path was dry-run on SYNTHETIC values (s5r_product.py --dry, look "
            "2026-10-17T03:50Z 'dryrun_product')"],
        "code_at_registration": hashes(code),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    doc = declaration_doc()
    if a.dry_run:
        errs = prereg3.validate_prereg(doc)
        print(json.dumps({"errors": errs, "inputs": len(doc["inputs"])}, indent=1))
        print(json.dumps({k: v for k, v in doc.items() if k not in ("inputs",
                                                                     "code_at_registration")},
                         indent=1, ensure_ascii=False)[:6000])
        return 0
    path, sha = prereg3.write_prereg(doc)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

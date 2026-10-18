"""WP S3 repair round 3 (S3r): write ``results/dataprep/reports/S3.repair.json`` from the evidence
files (numbers are read, not retyped).

    python -I dataprep_staging/s3_gnss/v3/repair_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import DATAPREP, REPORTS_DIR  # noqa: E402

EV = REPORTS_DIR / "S3_evidence" / "repair_2026-10-18"
EV_REL = "results/dataprep/reports/S3_evidence/repair_2026-10-18"


def r3(x):
    return None if x is None else round(float(x), 4)


def main() -> int:
    acc = EV / "acceptance"
    vp = json.loads((acc / "validation_preorigin.json").read_text())
    vf = json.loads((acc / "validation_full.json").read_text())
    tune = json.loads((EV / "tune_v3c" / "tune.json").read_text())
    post = json.loads((EV / "post_test_anatomy.json").read_text())
    pre_id = "S3.gnss_clean_acceptance.v3"
    prereg = json.loads((DATAPREP / "preregistration" / f"{pre_id}.json").read_text())
    idx = [json.loads(x) for x in (DATAPREP / "preregistration" / "INDEX.jsonl")
           .read_text().splitlines() if x.strip()]
    sha = [r["sha256"] for r in idx if r["id"] == pre_id][0]
    a1p, a1f = vp["A1_injected_steps"], vf["A1_injected_steps"]
    a2p, a2f = vp["A2_2021_extra"], vf["A2_2021_extra"]
    s2 = {(r["size_order"], r["local_half"]): r for r in tune["stage2_grid"]}
    ch = s2[(tune["chosen_size_order"], tune["chosen_local_half"])]
    s1 = tune["stage1"]["chosen"]

    def a2row(a2, k):
        s = a2[k]
        return {"n": s["n_evaluable"], "same_sign": r3(s["same_sign_fraction"]),
                "pooled_ratio": r3(s["pooled_amplitude_ratio"]), "pass": s["pass"]}

    acceptance = [
        {"criterion": "A1 injected steps >= 2 cm recovered within +-5 mm and +-3 d, WRA, "
                      "pre-origin v3 (gate; fresh units seed 707; S3.gnss_clean_acceptance.v3)",
         "threshold": ">= 0.90", "measured": f"{a1p['wra']['recovered']:.3f} (1,000 trials; "
         f"v2 0.863, v1 0.816)", "passed": bool(a1p["wra"]["pass"]),
         "units": "recovery rate (fraction of injected steps)"},
        {"criterion": "A1 injected steps, NGL, pre-origin v3 (gate; fresh units seed 707)",
         "threshold": ">= 0.90", "measured": f"{a1p['ngl']['recovered']:.4f} (480 trials; "
         f"v2 0.8625, v1 0.891)", "passed": bool(a1p["ngl"]["pass"]),
         "units": "recovery rate"},
        {"criterion": "A2 2021 extra survives ngl_offfan removal, pre-origin v3 (SECOND LOOK on "
                      "spent pair units; declared before scoring)",
         "threshold": "same sign >= 0.75 and pooled ratio >= 0.5",
         "measured": f"same sign {a2p['cm_removed_ngl_offfan']['same_sign_fraction']:.3f}, "
                     f"pooled ratio {a2p['cm_removed_ngl_offfan']['pooled_amplitude_ratio']:.3f} "
                     f"({a2p['cm_removed_ngl_offfan']['n_evaluable']} pairs); conditional, no "
                     "evidential weight (moved by GFES 2020 step handling)",
         "passed": bool(a2p["cm_removed_ngl_offfan"]["pass"]),
         "units": "fraction of pairs; ratio of pooled extras"},
        {"criterion": "A2, wra_hf removal, pre-origin v3 (SECOND LOOK on spent units)",
         "threshold": "same sign >= 0.75 and pooled ratio >= 0.5",
         "measured": f"same sign {a2p['cm_removed_wra_hf']['same_sign_fraction']:.3f}, pooled "
                     f"ratio {a2p['cm_removed_wra_hf']['pooled_amplitude_ratio']:.3f}; "
                     "conditional, no evidential weight",
         "passed": bool(a2p["cm_removed_wra_hf"]["pass"]),
         "units": "fraction of pairs; ratio of pooled extras"},
        {"criterion": "Gate (A1 both families on fresh units AND A2 both common modes)",
         "threshold": "all pass; A2-second-look passes make it conditional",
         "measured": "failed (A1 fails in both families)", "passed": False, "units": "boolean"},
        {"criterion": "A1 on the full records (reported, never gating, U7)",
         "threshold": ">= 0.90 (reported)",
         "measured": f"WRA {a1f['wra']['recovered']:.3f}, NGL {a1f['ngl']['recovered']:.3f} "
                     "(v2 0.819 / 0.868)", "passed": False, "units": "recovery rate"},
        {"criterion": "A1 secondary, harness family G r=20..29, pre-origin (reported)",
         "threshold": "reported only",
         "measured": "WRA {:.3f}, NGL {:.3f}".format(
             vp["A1_secondary_harness_family_G"]["wra"]["recovered"],
             vp["A1_secondary_harness_family_G"]["ngl"]["recovered"]),
         "passed": False, "units": "recovery rate"},
        {"criterion": "A3 excess-rate stability at co-located rings (report only)",
         "threshold": "delivered (no threshold in the plan)",
         "measured": "median excess {:.1f} mm/yr (2020-21) and {:.1f} (2023-02..2025-03)".format(
             vf["A3_excess_stability"]["median_pre_excess_mm_yr"],
             vf["A3_excess_stability"]["median_post_excess_mm_yr"]),
         "passed": True, "units": "mm/yr"},
        {"criterion": "Diagnosis written before any method change (repair discipline a)",
         "threshold": "S3.repair.md Part 1 before the first design run",
         "measured": "Part 1 written ~01:25 CST; first design run 01:26 CST; snapshot registered",
         "passed": True, "units": "procedure"},
        {"criterion": "Thresholds unchanged; re-test on fresh units (discipline c, d)",
         "threshold": "same thresholds; new seed never drawn",
         "measured": "thresholds as v1/v2; A1 seed 707 fresh (assert_can_score ok, 0 "
                     "post-registration looks); A2 labelled second look",
         "passed": True, "units": "procedure"},
        {"criterion": "Products as new versions with MANIFEST.json that re-hash",
         "threshold": "checker 0 failures for the new versions",
         "measured": "gnss_clean/{preorigin,full}/v3: 188 files, 0 failures; v1/v2 fail on the "
                     "upstream mechanism.py change (commit 8d51b46, not this round)",
         "passed": True, "units": "files"},
        {"criterion": "Loader guard refuses gated uses while the gate fails",
         "threshold": "13 probes as expected",
         "measured": "gated uses refused (GateNotPassed/forbidden); design_time_evidence (pre) "
                     "and after_the_fact_analysis open", "passed": True, "units": "probes"},
    ]
    out = {
        "wp": "S3r", "parent_wp": "S3", "status": "partial", "date": "2026-10-18",
        "gate_passed": False, "gate_status": "failed",
        "summary": (
            "Third registered test of the GNSS cleaning gate fails on fresh units: A1 WRA "
            f"{a1p['wra']['recovered']:.3f} and NGL {a1p['ngl']['recovered']:.3f} against 0.90 "
            "(v2 0.863 / 0.8625). The repaired detector removes the diagnosed mechanisms "
            "(miscalibrated, stale z; gap-blind ramp test; excursion artefacts; ERLN cap "
            "exhaustion) and reaches the data-limited regime: the step-size error sd of 2.9-3.2 "
            "mm, set by the noise of the daily series, leaves 7-9 % of steps of every size "
            "outside +-5 mm. A2 passes numerically only as a declared second look on spent "
            "units (conditional, no evidential weight); its failure in v1/v2 mostly reflected "
            "GNSS-ring physics (timing of whole-column vs ring-interval compaction at KTES and "
            "TKJS, present in raw vendor data) and step handling, not a common-mode defect, and "
            "no fresh co-located reference exists. gnss_clean v3 is built and stays excluded "
            "from gated uses."),
        "diagnosis": {
            "A1": [
                "D1 size estimate noise-limited (null sd 3.3 mm, 11.8 % beyond 5 mm; station "
                "correlation with trial failures 0.84/0.91); the free local slope doubles its "
                "variance and red noise makes longer windows worse",
                "D2 z noise scale mis-specified (AR(1), null z spread 0.83-1.54) and estimated "
                "once before detection (TKJS in-loop rho 0.69 vs 0.02)",
                "D3 sharpness test gap-unaware (NaN -> ramp rejection)",
                "D4 loop stops at a max-z candidate failing |D| >= delta_min",
                "D5 excursion test: artificial return at modelled steps; curvature read as a box",
                "D6 ERLN noisiest background (22 of 40 v2 failures); found at design time: its "
                "non-stationary noise exhausted the step cap under a station-wide scale"],
            "A2": ("decisive disagreements (KTES -1.2 vs +21 mm; TKJS -2.0 vs +14/+7.1 mm) exist in "
                   "the raw vendor data and come from the timing of the GNSS-minus-ring excess "
                   "(KTES +18 mm Aug-Sep 2020; TKJS abrupt GNSS-only drop 2020-11-05); the "
                   "pair set follows step handling at GFES/TKJS/YCES; removal shifts every site "
                   "by the mode's own extra (ngl_offfan -5.7 mm, within interannual sd 5.4 mm; "
                   "wra_hf -0.4 mm); no fresh co-located reference exists"),
        },
        "repairs": [
            "R1' level-difference step size (+-10 d chosen) on the series minus the global model "
            "and the detection CM (R1, a quadratic trend, withdrawn: mathematically empty for a "
            "centred step)",
            "R2 empirical-null z re-estimated each iteration; R2b time-local (+-365 d)",
            "R3 gap-aware sharpness test", "R4 |D|-ranked candidates",
            "R5 joint-step quadratic box test with variance-inflation scaling",
            "R6 5-nearest-station WRA detection common mode"],
        "design": {"runs": "run 1 stopped (wrong premise), run 2 (seeds 515/616) exposed R2b, "
                           "run 3 (seeds 525/626) final",
                   "chosen_params": tune["chosen_params"], "stage1_chosen": s1,
                   "stage2_chosen": ch, "reference": tune["reference"],
                   "reference_full": tune["reference_full"]},
        "retest": {"prereg_id": pre_id, "sha256": sha, "created": prereg["created"],
                   "tool": prereg["tool"], "fresh_units": prereg["fresh_units"],
                   "assert_can_score": vp["looks_logged"][0],
                   "A1_preorigin": {f: {"recovered": r3(a1p[f]["recovered"]), "n": a1p[f]["n"],
                                        "by_size": {k: r3(v["recovered"]) for k, v in
                                                    a1p[f]["by_size"].items()},
                                        "found_within_3d": r3(a1p[f]["found_within_3d"])}
                                    for f in ("wra", "ngl")},
                   "A1_full_reported": {f: r3(a1f[f]["recovered"]) for f in ("wra", "ngl")},
                   "A2_second_look_preorigin": {k: a2row(a2p, k) for k in
                                                ("cm_removed_ngl_offfan", "cm_removed_wra_hf",
                                                 "uncorrected")},
                   "A2_full_reported": {k: a2row(a2f, k) for k in
                                        ("cm_removed_ngl_offfan", "cm_removed_wra_hf")},
                   "post_test_anatomy": {f: {k: post[f"A1_{f}"][k] for k in
                                             ("not_found", "size_fail", "size_err_sd_mm")}
                                         for f in ("wra", "ngl")}},
        "acceptance": acceptance,
        "products": [
            {"path": "results/dataprep/gnss_clean/preorigin/v3/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/preorigin/v3/MANIFEST.json",
             "fit_window_end": "2022-12", "gate": "failed",
             "allowed_uses_effective": ["design_time_evidence", "after_the_fact_analysis"]},
            {"path": "results/dataprep/gnss_clean/full/v3/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/full/v3/MANIFEST.json",
             "fit_window_end": "2026-09", "gate": "failed",
             "allowed_uses_effective": ["after_the_fact_analysis"]}],
        "data_needed_if_failed": (
            "A1: lower-noise heights (sub-daily or re-processed solutions from raw RINEX with "
            "better troposphere modelling) and WRA equipment logs (step dates as priors), or a "
            "user decision on a tolerance matching the measured 2.9-3.2 mm size precision (new "
            "registration). A2: independent 2020-21 references within 1 km of GNSS antennas "
            "not used by the failed tests (new piles/extensometers, leveling ties to the "
            "antennas; the 2020-21 leveling is withheld for other WPs), WRA monument/antenna "
            "metadata for KTES, TKJS and GFES, and a user decision on whether a Dec-minus-Jan "
            "calendar statistic can test a common-mode effect"),
        "slips": [
            "looks design_null_level (~1 min) and erln_trace (~5 min) recorded after their "
            "computation (design units / design-time evidence only)",
            "R1 as stated in Part 1 was wrong; corrected at design time (Part 2)",
            "A2 numerical pass contradicted the stated expectation"],
        "issues_for_downstream": [
            "gnss_clean v1/v2 manifests fail the checker on hydrophysics/twin/mechanism.py "
            "(upstream commit 8d51b46, 2026-10-18 01:13 CST); S3 v2 registration lists it as "
            "a changed input",
            "drought-time steps undecidable without equipment logs: TKJS 2020-11-05 (CM-only, "
            "flag 3), HNES 2021-04-13 (own series, flag 1)",
            "gated uses (S4 off-fan reference, S1b, V5, twin GNSS holdout) stay blocked"],
        "looks": "18 S3 looks this round (harness.prereg3): 4 diagnosis, 2 smoke, 3 design-time "
                 "evidence, 4 design seeds, 1 registered score, 1 reported harness G, 1 A2 "
                 "second look, 1 full variant, 1 post-test diagnosis",
        "ledger": "results/dataprep/ledger/rows/S3.csv: +1 registration row, +759 rows "
                  "(report_v3.py); ledger check 0 conflicts",
        "evidence": EV_REL,
        "report_path": "results/dataprep/reports/S3.repair.md",
        "claude_api_tokens": 0, "gpu": "none", "downloads": "none",
    }
    (REPORTS_DIR / "S3.repair.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["retest"]["A1_preorigin"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

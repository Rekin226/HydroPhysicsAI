"""WP S3 repair round 2026-10-13: write ``results/dataprep/reports/S3.json`` (the WP result as it
now stands) from the v2 evidence, carrying the v1 and resume-round records forward.

    python -I dataprep_staging/s3_gnss/v2/report_json.py

Numbers are read from the evidence files (validation_*.json, repair_checks.json,
repair_rows.json, tune.json); the prose fields restate the report's Repair round section.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import DATAPREP, REPO  # noqa: E402

REP = DATAPREP / "reports"
EV = REP / "S3_evidence" / "repair_2026-10-13"


def r3(x):
    return None if x is None else round(float(x), 3)


def main() -> int:
    prev = json.loads((REP / "S3.r2.json").read_text())
    vp = json.loads((EV / "acceptance" / "validation_preorigin.json").read_text())
    vf = json.loads((EV / "acceptance" / "validation_full.json").read_text())
    rc = json.loads((EV / "repair_checks.json").read_text())
    rr = json.loads((EV / "repair_rows.json").read_text())
    tune = json.loads((EV / "tune_r3" / "tune.json").read_text())
    pre_path = DATAPREP / "preregistration" / "S3.gnss_clean_acceptance.v2.json"
    pre_sha = hashlib.sha256(pre_path.read_bytes()).hexdigest()
    a1, a1f = vp["A1_injected_steps"], vf["A1_injected_steps"]
    a2, a2f = vp["A2_2021_extra"], vf["A2_2021_extra"]
    a3 = vf["A3_excess_stability"]

    def bs(s):
        b = s["by_size"]
        return "/".join(f"{b[k]['recovered']:.3f}" for k in ("2cm", "3cm", "5cm", "10cm"))

    n_both = sum(1 for r in a3["table"] if r.get("pre_excess_mm_yr") is not None
                 and r.get("post_excess_mm_yr") is not None)
    n_post = sum(1 for r in a3["table"] if r.get("post_excess_mm_yr") is not None)
    n_pre = sum(1 for r in a3["table"] if r.get("pre_excess_mm_yr") is not None)
    chg = sorted(r["excess_change_mm_yr"] for r in a3["table"]
                 if r.get("excess_change_mm_yr") is not None)
    med_chg = (chg[len(chg) // 2] if len(chg) % 2 else
               0.5 * (chg[len(chg) // 2 - 1] + chg[len(chg) // 2])) if chg else None
    ck = rr["S3.repair.checker"]
    acceptance = [
        {"criterion": "A1 injected steps >= 2 cm recovered within +-5 mm and +-3 d, WRA, "
                      "pre-origin variant v2 (gate, S3.gnss_clean_acceptance.v2, seed 404)",
         "threshold": "recovery rate >= 0.90",
         "measured": f"{a1['wra']['recovered']:.3f} ({a1['wra']['n']} trials, 25 stations; "
                     f"2/3/5/10 cm {bs(a1['wra'])}; found within 3 d "
                     f"{a1['wra']['found_within_3d']:.3f}; size err p90 "
                     f"{a1['wra']['size_err_mm_p90']:.1f} mm). v1 (seed 303): 0.816",
         "passed": bool(a1["wra"]["pass"])},
        {"criterion": "A1 injected steps, NGL, pre-origin variant v2 (gate)",
         "threshold": "recovery rate >= 0.90",
         "measured": f"{a1['ngl']['recovered']:.3f} ({a1['ngl']['n']} trials, 12 stations; "
                     f"{bs(a1['ngl'])}; found {a1['ngl']['found_within_3d']:.3f}; p90 "
                     f"{a1['ngl']['size_err_mm_p90']:.1f} mm). v1: 0.891",
         "passed": bool(a1["ngl"]["pass"])},
        {"criterion": "A2 2021 extra compaction survives the removal of ngl_offfan (plan's "
                      "off-fan common mode), pre-origin v2 (gate; declared, not blind)",
         "threshold": "same sign >= 0.75 of evaluable pairs and pooled amplitude ratio >= 0.5",
         "measured": f"same sign {a2['cm_removed_ngl_offfan']['same_sign_fraction']:.2f}, "
                     f"pooled ratio {a2['cm_removed_ngl_offfan']['pooled_amplitude_ratio']:.3f}"
                     f" ({a2['cm_removed_ngl_offfan']['n_evaluable']} pairs). v1 (not gating "
                     "then): 0.571 / 0.349",
         "passed": bool(a2["cm_removed_ngl_offfan"]["pass"])},
        {"criterion": "A2, wra_hf removed, pre-origin v2 (gate; declared, not blind)",
         "threshold": "same sign >= 0.75 and pooled ratio >= 0.5",
         "measured": f"same sign {a2['cm_removed_wra_hf']['same_sign_fraction']:.2f}, pooled "
                     f"ratio {a2['cm_removed_wra_hf']['pooled_amplitude_ratio']:.3f} "
                     f"({a2['cm_removed_wra_hf']['n_evaluable']} pairs; uncorrected "
                     f"{a2['uncorrected']['same_sign_fraction']:.2f} / "
                     f"{a2['uncorrected']['pooled_amplitude_ratio']:.3f}). v1 (gating): "
                     "0.571 / 0.737",
         "passed": bool(a2["cm_removed_wra_hf"]["pass"])},
        {"criterion": "A3 excess-rate stability at co-located rings (report only; not an "
                      "acceptance criterion)",
         "threshold": "delivered (report only; no pass/fail threshold in the plan)",
         "measured": f"v2 full: median excess {a3['median_pre_excess_mm_yr']:.1f} mm/yr "
                     f"2020-21 ({n_pre} sites), {a3['median_post_excess_mm_yr']:.1f} mm/yr "
                     f"2023-02..2025-03 ({n_post} sites); {n_both} sites in both, median "
                     f"change {med_chg:.1f} mm/yr (v1: 21.1 / 18.8, 5 sites, -1.1)",
         "passed": True},
        {"criterion": "A1 on the full records (reported, never gating, U7)",
         "threshold": ">= 0.90 (reported)",
         "measured": f"v2 WRA {a1f['wra']['recovered']:.3f}, NGL {a1f['ngl']['recovered']:.3f}"
                     " (v1 0.805 / 0.860)",
         "passed": bool(a1f["pass"])},
        {"criterion": "Repair, major 1: pre-origin variant built with pre-origin evidence only "
                      "(no 2023+ evidence in station or common-mode membership; exemptions [] "
                      "accurate)",
         "threshold": "LNJS, YSLL, ANES not excluded on 2023-25 evidence; no post-origin-"
                      "motivated station in the pre-origin variant",
         "measured": "preorigin/v2: WRA 25 stations, LNJS/YSLL/ANES ordinary members; NGL 12 "
                     "(11 fan box + JSAN); reference PINT, PAKU, S103, GUKN, CHUA (pre-origin "
                     f"rule); exemptions {rc['S3.repair.preorigin.exemptions_v2']}",
         "passed": True},
        {"criterion": "Repair, major 2: A2 interpretation corrected; A2 tested for each offered "
                      "common mode; product cards quantify the ngl_offfan effect",
         "threshold": "delivered",
         "measured": "v1 recomputed: ngl_offfan removal 82.6 -> 45.4 mm (0.55; ratio 0.349); "
                     f"full/v2 82.6 -> {a2f['cm_removed_ngl_offfan']['pooled_gnss_extra_mm']:.1f}"
                     f" mm (ratio {a2f['cm_removed_ngl_offfan']['pooled_amplitude_ratio']:.3f});"
                     f" ngl_offfan own 2021 extra "
                     f"{a2['cm_own_extra_ngl_offfan']['extra_mm']:.1f} mm vs interannual sd "
                     f"{a2['cm_own_extra_ngl_offfan']['other_years_sd_mm']:.1f} mm; report "
                     "and v1 text corrected",
         "passed": True},
        {"criterion": "Repair, minor: detected steps re-tested after the final local size",
         "threshold": "0 detected steps with |size| < delta_min",
         "measured": f"pre-origin {rc['S3.repair.preorigin.detected_below_delta_min_v2']} "
                     f"(v1 {rc['S3.repair.preorigin.detected_below_delta_min_v1']}), full "
                     f"{rc['S3.repair.full.detected_below_delta_min_v2']} (v1 "
                     f"{rc['S3.repair.full.detected_below_delta_min_v1']}); "
                     f"{len(rc['S3.repair.preorigin.retest_dropped'])}/"
                     f"{len(rc['S3.repair.full.retest_dropped'])} steps dropped; full WRA "
                     f"flag-3 share {rc['S3.repair.full.wra.flag_shares_v1']['3']:.3f} -> "
                     f"{rc['S3.repair.full.wra.flag_shares_v2']['3']:.3f}",
         "passed": True},
        {"criterion": "Common mode from off-fan stations kept as a separate product; raw and "
                      "CM-removed scoring",
         "threshold": "delivered",
         "measured": "common_mode.parquet (ngl_offfan, wra_hf 45 d, wra_onfan_fullband), never "
                     "removed from value; frozen archive scored raw, vendor-uncleaned, "
                     "ngl_offfan- and wra_hf-removed (after-the-fact, full/v2)",
         "passed": True},
        {"criterion": "NGL download for the fan stations with SOURCE.json",
         "threshold": "about 10 MB, SOURCE.json per fetch",
         "measured": "22.2 MB (unchanged since round 1); 4 SOURCE.json re-hash ok, 0 orphans",
         "passed": True},
        {"criterion": "Products with MANIFEST.json (plan 2.3) that re-hash",
         "threshold": "checker 0 failures",
         "measured": f"{ck['manifests']} gnss_clean manifests (v1+v2) and {ck['sources']} "
                     f"SOURCE.json, {ck['files_hashed']} files, {ck['manifests_failed']} "
                     f"failures, {ck['orphans']} orphans",
         "passed": ck["manifests_failed"] == 0 and ck["sources_failed"] == 0},
        {"criterion": "v2 test registered before any look at its units (harness.prereg3)",
         "threshold": "assert_can_score ok with inputs verified",
         "measured": f"{rr['S3.repair.assert_can_score']}",
         "passed": rr["S3.repair.assert_can_score"].get("assert_can_score") == "ok"},
    ]
    gate_passed = bool(a1["wra"]["pass"] and a1["ngl"]["pass"] and a2["pass"])
    summary = (
        "Repair round after the independent verification of WP S3 (2026-10-13). A first repair "
        "attempt (night of 2026-10-12) wrote v2 code but was cut off before registering or "
        "building anything; this round reviewed and finished it, reran tune round r3 to "
        "completion from byte-identical code (the 12 configurations the cut-off run finished "
        "are reproduced exactly), registered a new test S3.gnss_clean_acceptance.v2 with "
        "harness.prereg3 (seed 404, same thresholds, A2 now required for each offered common "
        "mode), ran it and built gnss_clean/preorigin/v2 and gnss_clean/full/v2. Major 1 is "
        "fixed: the pre-origin variant keeps LNJS, YSLL and ANES (their pathology is in "
        "2023-25 only) and leaves out LSB0 and the regional stations chosen for post-2024 "
        "records, so its exemptions [] are accurate. Major 2 is fixed in the report and the "
        "products: wra_hf (a high-pass) cannot test the removal half of A2, while ngl_offfan "
        "removal cuts the pooled 2021 extra by about half (82.6 to 45.4 mm on v1, 42.2 mm on "
        "full/v2) because the off-fan median has its own 2021 extra of about -5.7 mm against "
        "an interannual sd of about 5 mm. The minor issues are fixed: steps are re-tested "
        "after their final size (0 detected steps below delta_min; full WRA flag-3 share "
        "18.4 % to 9.6 %), the wra_hf window follows its rule (45 d), ANES jump-cluster days "
        "carry flag 9 in the full variant, v2 manifests reference the registered v1 "
        "validate.py, and A3 is labelled report-only. The v2 gate failed again: A1 WRA 0.863 "
        "and NGL 0.863 against 0.90; A2 same sign 0.50 for both common modes. A2 moved from "
        "v1 mainly because the pair set changed (z_min 3.5 found common-mode-only steps at "
        "TKJS 2020-11-05 and YCES 2021-07-10 that remove them under U1, and GFES entered with "
        "a step-dominated -33.8 mm extra), not because of the common mode. Thresholds were "
        "not changed and no rescue variant was built. The guard refuses all four versions for "
        "gated uses; four user decision points are listed. 0 Claude API tokens, no GPU, "
        "nothing under hydrophysics/, tests/ or L0 written, nothing staged in git.")
    out = {
        "wp": "S3", "status": "partial", "date": "2026-10-13", "gate_passed": gate_passed,
        "summary": summary, "acceptance": acceptance,
        "prereg": {"current": {"id": "S3.gnss_clean_acceptance.v2",
                               "path": str(pre_path.relative_to(REPO)),
                               "created": json.loads(pre_path.read_text())["created"],
                               "sha256": pre_sha, "tool": "harness.prereg3",
                               "assert_can_score": rr["S3.repair.assert_can_score"]},
                   "v1": prev["prereg"]},
        "products": [
            {"path": "results/dataprep/gnss_clean/preorigin/v2/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/preorigin/v2/MANIFEST.json",
             "fit_window_end": "2022-12", "gate": "failed (S3.gnss_clean_acceptance.v2)",
             "allowed_uses_effective": ["design_time_evidence", "after_the_fact_analysis"],
             "status": "current"},
            {"path": "results/dataprep/gnss_clean/full/v2/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/full/v2/MANIFEST.json",
             "fit_window_end": "2026-09", "gate": "failed (v2 gate)",
             "allowed_uses_effective": ["after_the_fact_analysis"], "status": "current"},
        ] + [dict(p, status="superseded by v2" + (
            "; station set rests on 2023-25 evidence: use as after_the_fact_analysis only"
            if "preorigin" in p["path"] else "")) for p in prev["products"]],
        "code_paths": sorted(str(p.relative_to(REPO)) for p in HERE.glob("*.py"))
                      + prev["code_paths"],
        "key_numbers": {
            "A1_v2_preorigin": {"wra": r3(a1["wra"]["recovered"]),
                                "ngl": r3(a1["ngl"]["recovered"])},
            "A1_v2_full": {"wra": r3(a1f["wra"]["recovered"]), "ngl": r3(a1f["ngl"]["recovered"])},
            "A2_v2_preorigin": {k: {"same_sign": r3(a2[k]["same_sign_fraction"]),
                                    "pooled_ratio": r3(a2[k]["pooled_amplitude_ratio"]),
                                    "n": a2[k]["n_evaluable"]}
                                for k in ("cm_removed_ngl_offfan", "cm_removed_wra_hf",
                                          "uncorrected")},
            "A2_v2_full": {k: {"same_sign": r3(a2f[k]["same_sign_fraction"]),
                               "pooled_ratio": r3(a2f[k]["pooled_amplitude_ratio"])}
                           for k in ("cm_removed_ngl_offfan", "cm_removed_wra_hf",
                                     "uncorrected")},
            "tune_r3": {"chosen": tune["chosen"], "local_half": tune["chosen_local_half"],
                        "hp_window": tune["chosen_params"]["hp_window"],
                        "reference": tune["reference"],
                        "reference_full": tune["reference_full"]},
            "repair_checks": "results/dataprep/reports/S3_evidence/repair_2026-10-13/"
                             "repair_checks.json",
            "report_numbers": "results/dataprep/reports/S3_evidence/repair_2026-10-13/"
                              "report_numbers_v2.json",
            "v1": prev["key_numbers"]},
        "deviations_from_plan": prev["deviations_from_plan"] + [
            "Repair round: a second registered test (S3.gnss_clean_acceptance.v2, seed 404) of "
            "the same design family after the verifier's findings, disclosed as such in the "
            "registration; same thresholds; A2 made stricter (each offered common mode)",
            "Repair round: the pre-origin NGL family leaves out LSB0 (a plan candidate) and the "
            "regional continuity set as a precaution, because both were singled out with their "
            "post-2024 records; the full variant keeps them",
            "Repair round: the full variant's jump-cluster flag (ANES and others) was written "
            "after seeing 2023-25 data; full variant only",
            "Repair round: the v2 test used harness.prereg3 (A1 repair round 2), v1 used "
            "harness.prereg/prereg2"],
        "issues_for_downstream": [
            "Gate failed twice (v1 seed 303, v2 seed 404): the guard refuses gnss_clean "
            "preorigin/full v1 and v2 for datum_fit, calibration_target, scoring_target and "
            "validator; S4's off-fan reference, S1b, V5 and the twin GNSS holdout are blocked "
            "until a user decision",
            "Decision 1 (size tolerance): +-5 mm sits at the precision of a local step estimate; "
            "8-16 % of found 3-10 cm steps miss it in both versions; post-hoc +-10 mm gives "
            "0.943 (WRA) and 0.940 (NGL) in v2 (diagnostic)",
            "Decision 2 (A2): a GNSS-ring agreement test on 6-7 pairs whose composition follows "
            "from flag-3 exclusions; its removal half is vacuous for wra_hf and dominated by "
            "the off-fan reference's own interannual variation (~5 mm) for ngl_offfan; a "
            "revision needs a fixed pair set, unique-station counting, a flag-3 policy, a new "
            "registration id and new seeds (303 and 404 are spent)",
            "Decision 3 (common-mode-only steps): z_min 3.5 (declared rule) corrects steps seen "
            "only after removing the on-fan common mode, including TKJS 2020-11-05 -19.5 mm and "
            "YCES 2021-07-10 -15.3 mm inside the 2021 deformation signal; whether such steps "
            "are corrected (flag 3) or only flagged is a design decision",
            "Decision 4 (catalogue labelling): a detected step within 15 days of a catalogue "
            "date takes its label and skips the re-test (GFES 2020-06-27 +13.6 mm survives)",
            "gnss_clean/preorigin/v1 is selection-contaminated (station set from 2023-25 "
            "evidence); use preorigin/v2 for design-time evidence",
            "For S4: ngl_offfan is the independent reference but removes ~45-50 % of the pooled "
            "2021 extra at co-located sites; its network thins to 3-4 stations after 2024-04",
            "A3: with 7 sites in both windows the median excess change is -4.7 mm/yr (HNES "
            "-12.2, KTES -8.7, SSES -8.0); median excess 21.1 (2020-21) vs 15.0 mm/yr "
            "(2023-25); compare the twin through the excess, with several mm/yr uncertainty "
            "after 2023",
            "Flag-3 loss: full WRA 9.6 % (v1 18.4 %); 17 on-grid WRA stations keep >= 12 "
            "flag-0/1 months in 2023-25 (v1 15)",
            "WRA quality after 2023: 26 steps in 2023-01..2025-03 vs 7 in 2020-22 (full v2); "
            "jump clusters at ANES, SNES, KTES",
            "Provenance: keep every v2 code file listed in the v2 manifests unchanged, or the "
            "checker fails (build a v3 instead)"],
        "evidence": "results/dataprep/reports/S3_evidence/ (round 1: acceptance_spec.json, "
                    "tune/, tune_r2/, acceptance/; resume: resume_2026-10-12/; first repair "
                    "attempt: repair_r3/; repair round: repair_2026-10-13/)",
        "looks": prev["looks"] + "; repair round 2026-10-13: 8 looks with harness.prereg3 "
                 "(tune rerun, smoke build, A1 score, harness G, A2, full variant, 2 diagnose)",
        "ledger": "results/dataprep/ledger/rows/S3.csv (1,319 rows: 441 before the repair "
                  "round + 1 pre-registration + 785 + 72 + 20)",
        "claude_api_tokens": 0, "gpu": "none",
        "report_path": "results/dataprep/reports/S3.md",
        "resume_round": prev["resume_round"],
        "repair_round": {
            "date": "2026-10-13", "previous_report": "results/dataprep/reports/S3.r2.md / "
                                                     "S3.r2.json",
            "verifier_major": ["U7 station set of the pre-origin variant: fixed in "
                               "preorigin/v2", "A2 interpretation: corrected; A2 per common "
                               "mode in v2"],
            "verifier_minor": ["wra_hf window rule: applied (45 d)", "step re-test: added",
                               "ANES flag: jump-cluster flag 9 (full)", "registered v1 "
                               "validate.py: referenced from v2 manifests", "A3 labelled "
                               "report-only", "A3/A2 flag-3 losses: reduced by the re-test"],
            "evidence": "results/dataprep/reports/S3_evidence/repair_2026-10-13/"},
    }
    (REP / "S3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("gate_passed", gate_passed, "acceptance rows", len(acceptance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S1a repair round r3 (S1ar): ledger rows, the JSON report and a check of the quoted numbers.

    python -m s1_rings.r3_report            (from dataprep_staging/)

Reads only evidence files of this round and the v3 product; appends rows to
results/dataprep/ledger/rows/S1a.csv (new number ids ``S1a.r3.*``; one role per source) and
writes results/dataprep/reports/S1a.repair.json and
S1a_evidence/repair_r3/numbers_check.json (every number quoted in the report's headline,
recomputed from its evidence file).
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

R3 = K.EVIDENCE / "repair_r3"
REPORTS = K.REPO / "results" / "dataprep" / "reports"
V3 = K.REPO / "results" / "dataprep" / "rings_bridged" / "v3"
PRE2012_SITES = ("元長國小", "光復國小", "內寮派駐站", "嘉興國小", "土庫國中", "客厝國小",
                 "新生國小", "湖南國小", "溪州國小", "秀潭國小")


def j(p: Path) -> dict:
    return json.loads(p.read_text())


def ring(s: str) -> str:
    return rel_to_repo(K.COMPACTION_DIR / f"{s}.parquet")


def ledger_rows() -> list[dict]:
    rows = []
    reg_rows = rel_to_repo(K.EVIDENCE / "synthetic" / "synthetic_rows.parquet")
    pre_v1 = "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json"
    pre_t = "results/dataprep/preregistration/S1a.e_synthetic_fresh.v3.json"
    pre_r = "results/dataprep/preregistration/S1a.rings_bridged_rules.v3.json"
    dbm = rel_to_repo(K.DEEP_BENCH_V2)
    calib = rel_to_repo(R3 / "calibration.json")

    def add(nid, srcs, note):
        for src, role in srcs:
            rows.append({"number_id": nid, "source": src, "role": role,
                         "product_version": "rings_bridged/v3", "note": note})

    rings14 = [(ring(s), "input") for s in K.SITES]
    rings10 = [(ring(s), "input") for s in PRE2012_SITES]
    for nid, note in (
            ("S1a.r3.diag.forcing_full_rerun", "registered pipeline re-executed with and without "
                                               "forcing (pools recomputed), after the fact"),
            ("S1a.r3.diag.bdeep_error_field", "b_deep of the uninjected 2014-05..2021-12 surveys"),
            ("S1a.r3.diag.timing_deep_scale", "survey-to-survey depth scale of the deep rings"),
            ("S1a.r3.diag.line_calibration", "nominal vs empirical likelihood ratios, P(A) "
                                             "reliability, pools, worst-window anatomy")):
        add(nid, rings14 + [(dbm, "input"), (reg_rows, "input"), (pre_v1, "input")],
            note + " (spent units of S1a.e_synthetic_mixed.v1; diagnosis, not a test)")
    add("S1a.r3.calibration_sigma_x_sigma_c",
        rings14 + [(rel_to_repo(R3 / "diag" / "truth_bdeep.csv"), "fit_target")],
        "sigma_x 1.267 mm, sigma_c 1.349 mm from uninjected pseudo-gaps 2014-05..2021-12")
    fresh_src = rings10 + [(rel_to_repo(K.EVIDENCE / "assignment.json"), "input"),
                           (calib, "input"), (pre_t, "input"), (pre_r, "input")]
    add("S1a.r3.e_fresh_pre2012_r3", fresh_src,
        "GATING registered test S1a.e3.pre2012_r3_groupcond (worst-window RMSE)")
    add("S1a.r3.e_fresh_pre2012_controls_and_stress", fresh_src + [(dbm, "input")],
        "registered-method control and stress injections on the pre-2012 draws (reported)")
    add("S1a.r3.e_late2019", rings14 + [(dbm, "input"), (calib, "input"), (pre_t, "input")],
        "late-2019 start months (fresh start months, design-exposed surveys; reported)")
    add("S1a.r3.e_second_look_spent", rings14 + [(dbm, "input"), (calib, "input"),
                                                 (reg_rows, "input"), (pre_v1, "input")],
        "r3 method on the spent windows and seeds: SECOND LOOK, conditional, never gating")
    add("S1a.r3.retest_after_the_fact",
        [(rel_to_repo(R3 / "retest" / "fresh_rows.parquet"), "input"),
         (rel_to_repo(R3 / "retest" / "spent_second_look_rows.parquet"), "input"),
         (reg_rows, "input")], "after-the-fact analysis of the scored re-test")
    real = rings14 + [(calib, "input"), (pre_r, "input")]
    add("S1a.r3.v3_assignment_and_scales", real, "v3 group evidence, P(A), pooled scales")
    add("S1a.r3.v3_a_registered", real, "check (a), registered definition, v3 values")
    add("S1a.r3.v3_a_corrected", real, "corrected (a): SECOND LOOK on the 14 design-time sites")
    add("S1a.r3.v3_b_pile", real + [(dbm, "input")],
        "check (b); deep_benchmarks/v2 (gate failed) read as after_the_fact_analysis")
    add("S1a.r3.v3_residual_offset", real, "residual offset across the gap")
    add("S1a.r3.v3_path_and_flags", real + [(dbm, "input"),
                                            (rel_to_repo(K.PRINTED_V2), "validator")],
        "2022 path (pile-shaped where the feed check passes) and flag counts")
    add("S1a.r3.v3_vs_v2", real + [("results/dataprep/rings_bridged/v2", "input")],
        "v2 vs v3 corrections at the real sites")
    return rows


def numbers() -> dict:
    fr = j(R3 / "retest" / "fresh_result.json")["summaries"]
    sp = j(R3 / "retest" / "after_the_fact.json")["spent_second_look_paired"]["primary"]
    dg = j(R3 / "diag" / "summary.json")
    cal = j(R3 / "calibration.json")
    v = j(V3 / "validation.json")
    ff = dg["forcing_full_rerun"]["primary"]
    return {
        "e_fresh_r3_worst_cm": round(100 * fr["pre2012.r3.primary"]["worst_window_rmse_m"], 2),
        "e_fresh_r3_n_over": fr["pre2012.r3.primary"]["n_windows_over_1p5cm"],
        "e_fresh_r3_n_windows": fr["pre2012.r3.primary"]["n_windows"],
        "e_fresh_control_worst_cm": round(100 * fr["pre2012.registered.primary"]
                                          ["worst_window_rmse_m"], 2),
        "e_late2019_r3_worst_cm": round(100 * fr["late2019.r3.primary"]["worst_window_rmse_m"], 2),
        "e_late2019_control_worst_cm": round(100 * fr["late2019.registered.primary"]
                                             ["worst_window_rmse_m"], 2),
        "e_spent_second_look_worst_cm": round(sp["worst_r3_cm"], 2),
        "e_spent_registered_worst_cm": round(sp["worst_registered_cm"], 2),
        "e_spent_second_look_n_over": sp["n_over_r3"],
        "a_registered_v3_power": round(v["a_registered"]["power_25pct"], 3),
        "a_corrected_v3_power": round(v["a_corrected_second_look"]["power_25pct"], 3),
        "a_corrected_v3_z": round(v["a_corrected_second_look"]["pooled_z"], 2),
        "b_v3_pass": v["b_pile_implied_scale"]["pass"],
        "residual_v3": [v["residual_offset"]["n_within"], v["residual_offset"]["n"]],
        "sigma_x_mm": round(cal["sigma_x_mm"], 3), "sigma_c_mm": round(cal["sigma_c_mm"], 3),
        "forcing_rerun_max_row_change_cm": round(ff["max_abs_row_change_cm"], 2),
        "forcing_rerun_window_change_cm": [round(ff["window_rmse_change_cm_min"], 3),
                                           round(ff["window_rmse_change_cm_max"], 3)],
        "forcing_rerun_oracle_worst_cm": [round(x, 3) for x in ff["oracle_worst_cm"]],
        "bdeep_excess_mm_rms": dg["bdeep_error_truth"]["excess_mm_rms"],
        "bdeep_window_mean_excess_min_mm": dg["bdeep_error_truth"]["window_mean_excess_mm_min"],
    }


QUOTED = {   # the values written in S1a.repair.md / the summary
    "e_fresh_r3_worst_cm": 3.23, "e_fresh_r3_n_over": 11, "e_fresh_r3_n_windows": 16,
    "e_fresh_control_worst_cm": 2.94, "e_late2019_r3_worst_cm": 1.71,
    "e_late2019_control_worst_cm": 1.64, "e_spent_second_look_worst_cm": 2.08,
    "e_spent_registered_worst_cm": 2.79, "e_spent_second_look_n_over": 10,
    "a_registered_v3_power": 0.561, "a_corrected_v3_power": 0.748, "a_corrected_v3_z": -0.83,
    "b_v3_pass": True, "residual_v3": [13, 14], "sigma_x_mm": 1.267, "sigma_c_mm": 1.349,
    "forcing_rerun_max_row_change_cm": 5.27, "forcing_rerun_window_change_cm": [-0.275, 0.04],
    "forcing_rerun_oracle_worst_cm": [1.594, 1.512], "bdeep_excess_mm_rms": 2.92,
    "bdeep_window_mean_excess_min_mm": -3.62,
}


def report_json(nums: dict) -> dict:
    v = j(V3 / "validation.json")
    m = j(V3 / "MANIFEST.json")
    fr = j(R3 / "retest" / "fresh_result.json")
    return {
        "wp": "S1a", "round": "S1ar (repair round r3 of WP S1a)", "status": "partial",
        "date": _dt.date.today().isoformat(), "gate_passed": False,
        "gate_status": v["gate"]["status"],
        "previous_reports": ["results/dataprep/reports/S1a.r1.md", "results/dataprep/reports/S1a.md",
                             "results/dataprep/reports/S1a.json"],
        "report_path": "results/dataprep/reports/S1a.repair.md",
        "diagnosis_snapshot": {"path": "results/dataprep/reports/S1a_evidence/repair_r3/snapshots/"
                                       "S1a.repair.part1_diagnosis.md",
                               "sha256": "2fb453527f9bcaa1d43eceac508d7a8ad4c18c5db38832458ab2da253de52d43",
                               "utc": "2026-10-17T21:14:35Z"},
        "preregistrations": {
            "S1a.rings_bridged_rules.v3": "564d7b972e54dc4f2b4a377b2e76453f6506e6ca2a507da318487b2a337c8265",
            "S1a.e_synthetic_fresh.v3": fr["prereg_sha256"]},
        "numbers": nums,
        "products": [{"path": "results/dataprep/rings_bridged/v3/", "layer": "L2",
                      "gate_passed": False, "fit_window_end": m["fit_window_end"],
                      "allowed_uses": m["allowed_uses"],
                      "relation_to_v2": m["supersedes"]}],
        "allowed_uses_reason": v["allowed_uses_rule"]["reason"],
        "evidence": "results/dataprep/reports/S1a_evidence/repair_r3/ (diag/, calibration.json, "
                    "retest/, snapshots/, logs/, checker_rings_bridged.json, numbers_check.json); "
                    "results/dataprep/reports/S1a_evidence/checks_v3.json",
        "claude_api_tokens": 0, "gpu": "none", "downloads": "none",
        "code_paths": sorted(rel_to_repo(p) for p in (K.REPO / "dataprep_staging" / "s1_rings")
                             .glob("r3_*.py")) + ["dataprep_staging/s1_rings/tests/test_r3.py"],
    }


TEXT = {
    "summary": (
        "Repair round r3 of WP S1a. The diagnosis was written and hashed before any method "
        "change. The verifier's major issue was right: a full re-execution without the "
        "near-threshold forcing rule changes rows by up to 5.27 cm and window RMSEs by -0.275 to "
        "+0.040 cm, so r2's 'not a lever' claim is withdrawn. The (e) failure in 2016-07..2017-06 "
        "is a physical dry-season deep compaction pulse (2017 H1), not an instrument step. Its "
        "excess is a few mm per deep section, so the error of b_deep grows as 1/L_deep; the "
        "jackknife sd misses it, short deep sections gave confident wrong B votes, and the Xiutan "
        "pile line was miscalibrated. Five declared changes (length-aware b_deep sd, pile out of "
        "membership, empirical-Bayes prior, posterior-threshold pools, common-mode sd) were "
        "registered with a fresh re-test on pre-2012 windows never used before. The re-test FAILS: "
        "worst-window RMSE 3.23 cm against 1.5 cm (the registered method on the same draws, "
        "2.94 cm, scores lower mainly because its pools fall back to the injection mean). The "
        "late-2019 start months give 1.71 cm (control 1.64). As a second look on the spent windows "
        "the repair lowers the worst window from 2.79 to 2.08 cm but still fails. rings_bridged/v3 "
        "was built under the declared rules: (b) 5/5 now independent, residual offset 13/14, "
        "(a) power 0.561 registered and 0.748 corrected (second look), gate failed, "
        "after_the_fact_analysis only. No threshold was changed."),
    "acceptance": [
        {"criterion": "(e) fresh re-test S1a.e3.pre2012_r3_groupcond (gating)",
         "threshold": "<= 1.5 cm (post hoc)", "measured": "3.23 cm; 11 of 16 windows above",
         "passed": False},
        {"criterion": "(e) registered method on the same fresh draws (control, reported)",
         "threshold": "<= 1.5 cm (reported)", "measured": "2.94 cm; 9 of 16 above",
         "passed": False},
        {"criterion": "(e) late-2019 start months (reported; design-exposed surveys)",
         "threshold": "<= 1.5 cm (reported)", "measured": "r3 1.71 cm, control 1.64 cm",
         "passed": False},
        {"criterion": "(e) spent windows, SECOND LOOK (conditional, never gating)",
         "threshold": "<= 1.5 cm", "measured": "2.08 cm (registered 2.79); 10 of 42 above",
         "passed": False},
        {"criterion": "(a) registered definition with v3 values", "threshold":
         "|Z| < 2 and power >= 0.8", "measured": "Z 0.44, power 0.561", "passed": False},
        {"criterion": "(a) corrected, trend-based (SECOND LOOK)", "threshold":
         "|Z| < 2 and power >= 0.8", "measured": "Z -0.83, power 0.748", "passed": False},
        {"criterion": "(b) pile-implied scale (independent of membership in v3)", "threshold":
         "A +-25 %, B +-30 ppm", "measured": "5/5", "passed": True},
        {"criterion": "residual offset", "threshold": ">= 12 of 14 within +-1 cm",
         "measured": "13/14 (Yuanchang +1.58 cm)", "passed": True},
        {"criterion": "rings_bridged/v3 gate", "threshold": "all pass", "measured": "failed",
         "passed": False},
    ],
    "data_needed_if_failed": (
        "The WRA's record of the 2022 instrument change (plan R-d): per site the probe or cable "
        "used after the hiatus with its scale calibration and zero offset, ideally an overlap "
        "survey with both instruments. Otherwise an independent across-2022 compaction record "
        "at more ring sites (GNSS or InSAR products that pass their gates; deep benchmarks exist "
        "at only 4-5 sites). A non-exposed pseudo-gap era with long low-rate deep sections does "
        "not exist (2014-2021 spent, pre-2012 too short); post-2023 windows need a U7 ruling."),
    "issues_for_downstream": [
        "rings_bridged v1, v2 and v3 all open for after_the_fact_analysis only; ring_holdout, "
        "S5 and I1 cannot use them as validator or scoring target",
        "v3 is a newer rule set, not a validated improvement; v2 and v3 primary gap compaction "
        "differ by at most 0.71 cm (Yuanchang); S5 pins v2",
        "U7 reading for S1's post-gap checks still needs a user ruling",
        "S1b's checks (c) and (d) were scored against v2",
        "the observed-month sd remains an upper bound (r2 verifier minor 3)",
    ],
}


def main() -> None:
    if "--json-only" in sys.argv[1:]:
        p = REPORTS / "S1a.repair.json"
        out = j(p)
        out.update(TEXT)
        out["numbers"] = numbers()
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print("rewrote S1a.repair.json (text fields; ledger section kept)")
        return
    nums = numbers()
    mism = {k: {"quoted": QUOTED[k], "evidence": nums[k]} for k in QUOTED
            if (QUOTED[k] != nums[k] if not isinstance(QUOTED[k], float)
                else abs(QUOTED[k] - nums[k]) > 0.0051)}
    chk = {"n_checked": len(QUOTED), "mismatches": mism, "numbers": nums}
    (R3 / "numbers_check.json").write_text(json.dumps(chk, indent=1, ensure_ascii=False))
    print("numbers checked", len(QUOTED), "mismatches", mism)
    if mism:
        raise SystemExit("quoted numbers differ from the evidence")
    rows = ledger_rows()
    p = ledger.append_rows("S1a", rows)
    allr = ledger.read_all()
    bad = ledger.conflicts(allr)
    print("ledger rows appended", len(rows), "->", p.name, "| all fragments", len(allr),
          "conflicts", len(bad))
    out = report_json(nums)
    out["ledger"] = {"fragment": "results/dataprep/ledger/rows/S1a.csv", "rows_appended": len(rows),
                     "rows_all_fragments": len(allr), "conflicts": len(bad)}
    (REPORTS / "S1a.repair.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote S1a.repair.json")


if __name__ == "__main__":
    main()

"""WP S3 fix round of the S3r repair (2026-10-18): write ``results/dataprep/reports/S3.repair.json``
for the fix round from the evidence files (numbers are read, not retyped). The S3r-round JSON is
archived read-only in ``fix_round/S3.repair.s3r_round.json`` and referenced here.

    python -I -B dataprep_staging/s3_gnss/v4/repair_json_v4.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import REPORTS_DIR  # noqa: E402

EV = REPORTS_DIR / "S3_evidence" / "repair_2026-10-18"
FR = EV / "fix_round"
FR_REL = "results/dataprep/reports/S3_evidence/repair_2026-10-18/fix_round"


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    n = json.loads((FR / "report_numbers_v4.json").read_text())
    old = json.loads((FR / "S3.repair.s3r_round.json").read_text())
    reg = json.loads((EV / "acceptance" / "validation_preorigin.json").read_text())
    regf = json.loads((EV / "acceptance" / "validation_full.json").read_text())
    v4p = json.loads((FR / "acceptance_v4" / "validation_preorigin.json").read_text())
    v4f = json.loads((FR / "acceptance_v4" / "validation_full.json").read_text())
    a1r, a1p, a1f = reg["A1_injected_steps"], v4p["A1_injected_steps"], v4f["A1_injected_steps"]
    a2p, a2f = v4p["A2_2021_extra"], v4f["A2_2021_extra"]
    ms = json.loads((FR / "matched_sets.json").read_text())
    own2, own3 = (json.loads((REPORTS_DIR.parent / "gnss_clean" / "full" / v / "validation.json")
                             .read_text())["this_variant"]["A3_excess_stability"]
                  for v in ("v2", "v3"))
    chk = json.loads((FR / "checker_gnss_clean_v4.json").read_text())
    gp = json.loads((FR / "guard_probe_v4.json").read_text())
    ngl_same = n["S3.v4.A1.preorigin.ngl.same_trials_as_v3"]
    wsame = n["S3.v4.A1.preorigin.wra.same_trials_as_v3"]
    sm = n["S3.fix.smoke.r7b.full.composition_abs_mm"]
    st = n["S3.fix.smoke.r7b.full.steps_v3_v4"]
    f01 = n["S3.v4.flag01_days_2023_25"]["v4"]
    snaps = {nm: sha(FR / nm) for nm in ("S3.repair.parts1-7.4.snapshot.md",
                                         "S3.repair.parts1-7.5.snapshot.md",
                                         "S3.repair.parts1-7.5.corrected.snapshot.md")}

    def a2row(a2, k):
        s = a2[k]
        return {"n": s["n_evaluable"], "same_sign": r(s["same_sign_fraction"]),
                "pooled_ratio": r(s["pooled_amplitude_ratio"]), "pass": s["pass"]}

    a3m = ms["A3"]
    tw = ms["twin_raw_cleaned"]
    v4_checks = [m for m in chk["manifests"] if m["version"] == "v4"]
    acceptance = [
        {"criterion": "A1 injected steps >= 2 cm within +-5 mm and +-3 d, WRA, pre-origin "
                      "(gate; registered v3 test on fresh units seed 707; v4 adds a second look)",
         "threshold": ">= 0.90",
         "measured": f"registered v3: {a1r['wra']['recovered']:.3f}; v4 second look on the "
                     f"spent seed-707 units: {a1p['wra']['recovered']:.3f} (conditional, no "
                     f"evidential weight)",
         "passed": False, "units": "recovery rate"},
        {"criterion": "A1 WRA, pre-origin, v4 SECOND LOOK on the spent seed-707 units "
                      "(declared before the run; cannot open the gate)",
         "threshold": ">= 0.90",
         "measured": f"{a1p['wra']['recovered']:.3f} (exactly at the threshold: one trial "
                     f"fewer would fail; +3 trials vs v3: -3 on {wsame['n_same_position']} identical "
                     f"positions, +6 on {1000 - wsame['n_same_position']} re-drawn YSLL/FRES "
                     "positions); conditional (second look), no evidential weight",
         "passed": bool(a1p["wra"]["recovered"] >= 0.90), "units": "recovery rate"},
        {"criterion": "A1 injected steps, NGL, pre-origin (gate; registered v3 test on fresh "
                      "units; the v4 NGL family is identical to v3)",
         "threshold": ">= 0.90",
         "measured": f"registered v3: {a1r['ngl']['recovered']:.4f}; v4 re-run identical "
                     f"({a1p['ngl']['recovered']:.4f}, {ngl_same['changed_outcome']} changed "
                     f"trial outcomes)",
         "passed": False, "units": "recovery rate"},
        {"criterion": "A2 2021 extra survives ngl_offfan removal, pre-origin v4 (SECOND LOOK on "
                      "spent pair units)",
         "threshold": "same sign >= 0.75 and pooled ratio >= 0.5",
         "measured": f"same sign {a2p['cm_removed_ngl_offfan']['same_sign_fraction']:.3f}, "
                     f"pooled ratio {a2p['cm_removed_ngl_offfan']['pooled_amplitude_ratio']:.3f}"
                     f" ({a2p['cm_removed_ngl_offfan']['n_evaluable']} pairs); conditional "
                     "(second look), no evidential weight",
         "passed": bool(a2p["cm_removed_ngl_offfan"]["pass"]),
         "units": "fraction of pairs; ratio of pooled extras"},
        {"criterion": "A2, wra_hf removal, pre-origin v4 (SECOND LOOK on spent pair units)",
         "threshold": "same sign >= 0.75 and pooled ratio >= 0.5",
         "measured": f"same sign {a2p['cm_removed_wra_hf']['same_sign_fraction']:.3f}, pooled "
                     f"ratio {a2p['cm_removed_wra_hf']['pooled_amplitude_ratio']:.3f}; "
                     "conditional (second look), no evidential weight",
         "passed": bool(a2p["cm_removed_wra_hf"]["pass"]),
         "units": "fraction of pairs; ratio of pooled extras"},
        {"criterion": "Gate (A1 both families on fresh units AND A2 both common modes)",
         "threshold": "all pass on fresh units",
         "measured": "failed: A1 fails in both families on fresh units (v3); the v4 NGL family "
                     "is identical, so the NGL failure applies to v4; no fresh units remain",
         "passed": False, "units": "boolean"},
        {"criterion": "A1 on the full records (reported, never gating, U7; v4 second look)",
         "threshold": ">= 0.90 (reported)",
         "measured": f"v4 WRA {a1f['wra']['recovered']:.3f}, NGL {a1f['ngl']['recovered']:.3f} "
                     f"(v3 {regf['A1_injected_steps']['wra']['recovered']:.3f} / "
                     f"{regf['A1_injected_steps']['ngl']['recovered']:.3f})",
         "passed": False, "units": "recovery rate"},
        {"criterion": "A3 excess-rate stability at co-located rings (plan: reported; NO "
                      "threshold, so no pass value; listed as not passed)",
         "threshold": "none (report only)",
         "measured": "v4 own set: median |change| "
                     f"{v4f['A3_excess_stability']['median_abs_change_mm_yr']:.1f} mm/yr "
                     f"({a3m['own']['v4']['n']} sites; v3 "
                     f"{own3['median_abs_change_mm_yr']:.1f} ({a3m['own']['v3']['n']}), "
                     f"v2 {own2['median_abs_change_mm_yr']:.1f} ({a3m['own']['v2']['n']})); "
                     "sites common to v2 and v4 (6): v2 "
                     f"{a3m['common_v2_v4']['v2']['median_abs_change_mm_yr']:.1f}, v4 "
                     f"{a3m['common_v2_v4']['v4']['median_abs_change_mm_yr']:.1f}; common to "
                     f"v2-v4 ({a3m['common_v2_v3_v4']['v4']['n']}): v2 "
                     f"{a3m['common_v2_v3_v4']['v2']['median_abs_change_mm_yr']:.1f}, v3 "
                     f"{a3m['common_v2_v3_v4']['v3']['median_abs_change_mm_yr']:.1f}, v4 "
                     f"{a3m['common_v2_v3_v4']['v4']['median_abs_change_mm_yr']:.1f}",
         "passed": False, "units": "mm/yr"},
        {"criterion": "D7 fixed: no common-mode-only step, excursion or transient at a "
                      "neighbour-composition event in full/v4 (smoke check, declared 7.4)",
         "threshold": "the four named steps and the KTES/HNES/FRES/HLES composition flags gone; "
                      "full-composition days identical to v3",
         "measured": f"gone (full steps {st[0]} -> {st[1]}; flag-0/1 days 2023-25 HLES "
                     f"{f01['HLES']}, ANES {f01['ANES']}, HNES {f01['HNES']}, YCES "
                     f"{f01['YCES']}); max |v4 - v3| on full-composition days "
                     f"{n['S3.fix.smoke.r7b.full.max_abs_diff_full_composition_days_m']}; "
                     f"composition parts >= 10 mm {sm['v3']['n_ge_10']} -> {sm['v4']['n_ge_10']}"
                     f" (max {sm['v3']['max']} -> {sm['v4']['max']} mm; the remaining one "
                     "against a 1-station reference). First design R7 withdrawn at its smoke "
                     "check",
         "passed": True, "units": "steps / days / mm"},
        {"criterion": "Diagnosis before method change, thresholds unchanged, second looks "
                      "labelled (discipline a-d)",
         "threshold": "procedure",
         "measured": "Part 7.1-7.4 written 03:45 CST before any v4 method code; 7.5 (R7 flaw, "
                     "R7') at 03:52 before R7' was coded; snapshots " + ", ".join(
                         f"{k} {v[:12]}" for k, v in snaps.items()) + "; no threshold "
                     "changed; v4 A1/A2 labelled second looks; no new registered test",
         "passed": True, "units": "procedure"},
        {"criterion": "New product versions re-hash (v4)",
         "threshold": "checker 0 failures for v4",
         "measured": f"{len(v4_checks)} v4 manifests, "
                     f"{sum(m['n_hashed'] for m in v4_checks)} files, "
                     f"{sum(m['status'] != 'ok' for m in v4_checks)} failures",
         "passed": all(m["status"] == "ok" for m in v4_checks), "units": "files"},
        {"criterion": "Loader guard refuses gated uses (v4)",
         "threshold": "13 probes as expected",
         "measured": "open: " + ", ".join(f"{g['product'].split('/')[1]}:{g['use']}" for g in gp
                                          if g.get("ok")),
         "passed": [g["use"] for g in gp if g.get("ok")] == [
             "design_time_evidence", "after_the_fact_analysis", "after_the_fact_analysis"],
         "units": "probes"},
    ]
    out = {
        "wp": "S3r", "parent_wp": "S3", "round": "S3r fix round (after the verifier's result on "
                                                 "the S3r round)",
        "status": "partial", "date": "2026-10-18", "gate_passed": False, "gate_status": "failed",
        "summary": None,                 # filled below
        "s3r_round": {"archived_json": f"{FR_REL}/S3.repair.s3r_round.json",
                      "sha256": sha(FR / "S3.repair.s3r_round.json"),
                      "corrections": [
                          "A3 was listed as passed; it has no threshold (report only)",
                          "A3 text omitted its own statistic: median |change| 11.1 mm/yr in "
                          "v3 vs 4.6 in v2, on different site sets",
                          "twin R2 -3.60 -> -0.40 was mostly a change of station set",
                          "'better-calibrated step se and flags' did not hold for full/v3",
                          "summary: TKJS attributed to timing (Part 1.4: GNSS-only drop, "
                          "likely instrumental); 'data-limited' not established (only OLS "
                          "estimators tried); '7-9 % of every size' is 4.4-12.5 %",
                          "timestamps: Part 1 01:20:13 (not ~01:25), 2.6 02:37:22, Part 3 "
                          "02:58:20; slips: structure check 3.5 min (not ~1 min) before its "
                          "look, ERLN trace look 6 min after output / 10.5 min after start"]},
        "fix_round": {
            "verifier": {"critical": 0, "major": 1, "minor": 6, "violations": 0},
            "diagnosis_D7": {
                "mechanism": "v3 R6 detection CM = median over whichever of the 5 nearest WRA "
                             "members report; neighbour residual levels differ by cm over "
                             "2020-25, so outages move the median (Jan 2023 network outage)",
                "full_events": n["S3.fix.diag.events.full.composition_abs_mm"],
                "full_events_2020_22": n["S3.fix.diag.events.full.2020_22.composition_abs_mm"],
                "preorigin_events": n["S3.fix.diag.events.preorigin.composition_abs_mm"],
                "cm_only_steps_full_v3": n["S3.fix.diag.cm_only_steps.full"],
                "flag01_days_2023_25_v2_v3": {k.split(".")[-1]: v for k, v in n.items()
                                              if k.startswith("S3.fix.diag.flag01_days")},
                "ngl": {"preorigin": n["S3.fix.diag.ngl.preorigin"],
                        "full": n["S3.fix.diag.ngl.full"], "verdict": "no defect of this kind"},
                "product_cms": n["S3.fix.diag.product_cm_edges"],
                "preorigin_gate_affected": False},
            "repairs": [
                "R7 (withdrawn at its smoke check, never a product): absent neighbours imputed "
                "with offsets estimated against the current CM; a CM box longer than half the "
                "+-30 d offset window was absorbed (HLES Jan 2023 excursion +36.7 mm stayed)",
                "R7' (final): composition-invariant 5-nearest median; v3 value when all 5 "
                "report; absent neighbours imputed from PAIRWISE offsets (running medians of "
                "r_j - r_k over +-win days, >= min_obs common days; least squares, sum zero); "
                "no-neighbour days interpolated in time; win/min_obs are existing tuned values",
                "NGL detection CM, product CMs, detector, tuned parameters, station sets, "
                "flags, basis and transient rules, thresholds: unchanged",
                "__pycache__ removed from s3_gnss/, v2/, v3/; every run with python -I -B; all "
                "files under s3_gnss/ scanned for home paths"],
            "smoke_r7_withdrawn": {k.split("r7_withdrawn.")[1]: v for k, v in n.items()
                                   if k.startswith("S3.fix.smoke.r7_withdrawn.")},
            "smoke_r7b": {k.split("r7b.")[1]: v for k, v in n.items()
                          if k.startswith("S3.fix.smoke.r7b.")},
            "v4_second_looks": {
                "A1_preorigin": {"wra": r(a1p["wra"]["recovered"]),
                                 "ngl": r(a1p["ngl"]["recovered"]),
                                 "by_size_wra": {k: r(x["recovered"]) for k, x in
                                                 a1p["wra"]["by_size"].items()},
                                 "same_trials_as_v3": {
                                     "wra": n["S3.v4.A1.preorigin.wra.same_trials_as_v3"],
                                     "ngl": ngl_same}},
                "A1_full": {"wra": r(a1f["wra"]["recovered"]), "ngl": r(a1f["ngl"]["recovered"])},
                "A1_harness_G": {v: n.get(f"S3.v4.A1G.{v}.wra.recovered") for v in
                                 ("preorigin", "full")},
                "A2_preorigin": {k: a2row(a2p, k) for k in
                                 ("cm_removed_ngl_offfan", "cm_removed_wra_hf", "uncorrected")},
                "A2_full": {k: a2row(a2f, k) for k in ("cm_removed_ngl_offfan",
                                                       "cm_removed_wra_hf")},
                "label": "second looks on spent units: conditional, no evidential weight"},
            "traces": {f.name: json.loads(f.read_text())
                       for f in sorted((FR / "traces").glob("*.json"))},
            "after_the_fact_full_v4": {"A3_matched": {k: a3m[k] for k in
                                                      ("own", "common_v2_v3_v4",
                                                       "common_v2_v4")},
                                       "twin_raw_cleaned_matched": {
                                           k: {v: {"n": d["n"],
                                                   "r2": r(d["model_archive"]["r2"], 3),
                                                   "rmse": r(d["model_archive"]["rmse"], 3),
                                                   "bias": r(d["model_archive"]["bias"], 3)}
                                               for v, d in tw[k].items()}
                                           for k in ("own", "common_v2_v3_v4",
                                                     "common_v2_v4")},
                                       "stations": tw["stations"]},
        },
        "acceptance": acceptance,
        "products": [
            {"path": "results/dataprep/gnss_clean/preorigin/v4/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/preorigin/v4/MANIFEST.json",
             "fit_window_end": "2022-12", "gate": "failed",
             "allowed_uses_effective": ["design_time_evidence", "after_the_fact_analysis"]},
            {"path": "results/dataprep/gnss_clean/full/v4/", "layer": "L1",
             "manifest": "results/dataprep/gnss_clean/full/v4/MANIFEST.json",
             "fit_window_end": "2026-09", "gate": "failed",
             "allowed_uses_effective": ["after_the_fact_analysis"]}],
        "superseded": ["results/dataprep/gnss_clean/preorigin/v3/",
                       "results/dataprep/gnss_clean/full/v3/ (D7 composition artefacts in "
                       "2023-25; do not use)"],
        "report_path": "results/dataprep/reports/S3.repair.md (Part 7)",
        "evidence": FR_REL,
        "claude_api_tokens": 0, "gpu": "none", "downloads": "none",
        "data_needed_if_failed": (
            "A1: (1) an INDEPENDENT test needs new backgrounds (GNSS series or windows that "
            "neither the design's null evaluation nor any test has touched); none exists in "
            "these families; (2) before calling the size floor data-limited, an estimator "
            "study (noise-model GLS or power-law-plus-white-noise step estimates; the size "
            "error grows with step size, 4.4-12.5 % of found steps beyond 5 mm); (3) lower-noise "
            "solutions (re-processed RINEX, better troposphere) and WRA equipment logs; (4) or "
            "a user decision on a tolerance matching the measured size precision (new "
            "registration). " + old["data_needed_if_failed"].split("A2:", 1)[1].join(["A2:", ""])
            if "A2:" in old["data_needed_if_failed"] else old["data_needed_if_failed"]),
        "issues_for_downstream": old["issues_for_downstream"] + [
            "full/v3 carries composition artefacts in 2023-25 (D7): use full/v4",
            "other WPs' and the shared common/, harness/ __pycache__ directories under "
            "dataprep_staging/ hold .pyc files with absolute paths (Rule 4); not this WP's "
            "to delete"],
    }
    out["summary"] = (
        "Fix round: the verifier's major issue is confirmed and fixed. The v3 R6 detection "
        "common mode (median of the 5 nearest WRA members over whichever report) jumped by up "
        "to 55 mm when neighbours stopped or resumed (January 2023 outage), creating "
        "common-mode-only steps, excursions and flag-3 records in full/v3 and distorting A3 "
        "and the twin scoring. R7' makes it composition-invariant (identical to v3 on "
        "full-composition days; absent neighbours imputed from pairwise offsets); a first "
        "design (R7) was withdrawn at its smoke check. gnss_clean v4 is built; the pre-origin "
        "change is small and NGL is identical, so the gate stays failed on fresh units "
        f"(registered v3: WRA {a1r['wra']['recovered']:.3f}, NGL {a1r['ngl']['recovered']:.3f}); "
        f"v4 second looks: WRA {a1p['wra']['recovered']:.3f}, NGL "
        f"{a1p['ngl']['recovered']:.3f}. A3 is report only; on matched sites the v4 numbers "
        "replace the v3 ones. Minor items corrected (fresh-unit caveat, slip timings, "
        "timestamps, A3 pass value, summary overreach, __pycache__).")
    (REPORTS_DIR / "S3.repair.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                           default=str))
    print("wrote", REPORTS_DIR / "S3.repair.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

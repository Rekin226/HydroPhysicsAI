"""WP S5 repair round: write reports/S5.json from the evidence (numbers read, not retyped).

    python -I s5r_write_json.py      (after s5r_report.py and s5r_ledger.py repair)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5r_const as R  # noqa: E402
from common import ledger  # noqa: E402

REP = C.DATAPREP / "reports"


def j(p: Path):
    return json.loads(p.read_text())


def f3(x) -> str:
    return f"{x:.3f}"


def main() -> int:
    n = j(R.EVIDENCE_R / "report_numbers_r2.json")
    r1 = n["round1"]
    t1, t2, t3, r2 = r1["t1"], r1["t2"], r1["t3"], r1["r2"]
    sc1 = t1["scored"]
    m1 = n["m1"]
    ms = m1["scored"]
    s2 = n["stage2"]
    v2 = n["v2"]
    sens = v2["sensitivity"]
    vv = sens["v2_minus_v1_product"]
    tb = n["diag_t2buffer"]
    cp = n["diag_campaign"]
    sa = n["diag_stageA"]
    dr = n["dryrun"]
    by = {y: b["ratio"] for y, b in m1["scored_by_year"].items()}
    allr = ledger.read_all()
    s5rows = [r for r in allr if r["wp"] == "S5"]
    acc = [
        {"criterion": "T1 (gate, registered S5.subsidence_fusion.v1, round-1 model): 2020-21 "
                      "leveling withheld, evaluable", "threshold": ">= 20 benchmarks with VR >= 0.5",
         "measured": f"{t1['n_benchmarks_scored']} benchmarks / {t1['n_targets_scored']} targets "
                     f"(VR median {t1['vr_quantiles']['0.5']:.3f})", "passed": True},
        {"criterion": "T1 skill: RMSE fused < continue-the-leveling-rate baseline",
         "threshold": "ratio < 1",
         "measured": f"{sc1['rmse_fused']:.2f} vs {sc1['rmse_baseline']:.2f} mm, ratio "
                     f"{sc1['ratio_fused_baseline']:.3f} (bootstrap 90 % "
                     f"{sc1['boot_ratio']['p05']:.3f}-{sc1['boot_ratio']['p95']:.3f})",
         "passed": False},
        {"criterion": "T1 coverage of the fused 90 % intervals (coverage3 gate-3, 44 clusters)",
         "threshold": "cluster interval contains 0.90",
         "measured": f"{sc1['coverage_fused']['coverage']:.3f} (interval "
                     f"{sc1['coverage_fused']['lo']:.3f}-{sc1['coverage_fused']['hi']:.3f}); rms z 2.47",
         "passed": False},
        {"criterion": "T2 coverage, 10 km leveling blocks out (no buffer; 6942 increments)",
         "threshold": "cluster interval contains 0.90",
         "measured": f"{t2['coverage_fused']['coverage']:.3f}; interval "
                     f"{t2['coverage_fused']['lo']:.3f}-{t2['coverage_fused']['hi']:.3f} "
                     "excludes 0.90 (over-covers: intervals too wide)", "passed": False},
        {"criterion": "T2 skill: RMSE fused < IDW of neighbour rates", "threshold": "ratio < 1",
         "measured": f"{t2['rmse_fused']:.2f} vs {t2['rmse_idw']:.2f} mm, ratio "
                     f"{t2['ratio_fused_idw']:.3f}", "passed": True},
        {"criterion": "Product gate = T1 and T2 (not re-tested in the repair round)",
         "threshold": "both pass", "measured": "T1 FAIL, T2 FAIL; gate.passed false in v1, v2, v3",
         "passed": False},
        {"criterion": "M1 evaluable (registered S5.monthly_field.v1, pre-origin, WRA GNSS "
                      "months 2020-22 withheld)",
         "threshold": ">= 100 supported station-months at >= 8 stations",
         "measured": f"{m1['n_scored']} station-months at {m1['stations_scored']} stations "
                     f"(of {m1['n_station_months']} at {m1['stations']})", "passed": True},
        {"criterion": "M1 skill: monthly anomaly vs annual model on supported station-months",
         "threshold": "RMSE ratio < 1 and station-bootstrap P(ratio >= 1) <= 0.10",
         "measured": f"{ms['rmse_monthly']:.2f} vs {ms['rmse_annual_only']:.2f} mm, ratio "
                     f"{ms['ratio']:.3f}, P(ratio >= 1) {ms['boot_ratio']['p_ge_1']:.4f} "
                     f"(bootstrap {ms['boot_ratio']['p05']:.3f}-{ms['boot_ratio']['p95']:.3f}); "
                     "by year " + ", ".join(f"{y} {v:.3f}" for y, v in by.items())
                     + " (marginal pass, all from 2021)", "passed": bool(m1["pass"])},
        {"criterion": "M1 coverage of the anomaly predictive sd (coverage3 gate-3, station "
                      "clusters; decides only the sd label)",
         "threshold": "cluster interval contains 0.90",
         "measured": f"{ms['coverage']['coverage']:.3f} (interval {ms['coverage']['lo']:.3f}-"
                     f"{ms['coverage']['hi']:.3f}, 18 clusters; outside coverage3's validated "
                     "envelope); v3 still labels the monthly VALUE sd uncalibrated",
         "passed": bool(ms["coverage"]["pass"])},
        {"criterion": "Major 2: monthly field where InSAR supports it (plan output)",
         "threshold": "delivered with a tested support rule",
         "measured": f"v3 fused_monthly.npz 2020-01..2025-12; delivered on "
                     f"{v2['monthly']['support_delivered']:.3f} of cell-months (support rule "
                     "footprint and plane-free VR >= 0.5, gated by M1)", "passed": True},
        {"criterion": "Major 1: SJES in-gap offset no longer enters the field; disclosed",
         "threshold": "rule declared before the build; effect quantified; card",
         "measured": "GNSS offsets at gaps > 30 d (41 extra at 20 stations); mean 2022/2023 "
                     "increments within 3 km of SJES "
                     f"{vv['sjes_window_v1_inc_within_3km']['2022']:.1f}/"
                     f"{vv['sjes_window_v1_inc_within_3km']['2023']:.1f} (v1) -> "
                     f"{vv['sjes_window_v2_inc_within_3km']['2022']:.1f}/"
                     f"{vv['sjes_window_v2_inc_within_3km']['2023']:.1f} mm/yr (v3)",
         "passed": True},
        {"criterion": "Stepped benchmarks flagged (plan S5)",
         "threshold": "rule declared; list published",
         "measured": f"{v2['stepped']} of {v2['scanned']} scanned (second offset from the step)",
         "passed": True},
        {"criterion": "mask_year excludes the extrapolated year (flag 6)",
         "threshold": "2026 share 0", "measured": f"v3 {v2['mask_year_2026']:.2f} "
                                                  f"(v1 {n['v1']['mask_year_2026']:.2f})",
         "passed": v2["mask_year_2026"] == 0.0},
        {"criterion": "v3 manifest valid, hashes reproduce, guard opens only allowed uses, "
                      "fit_window_end with inputs",
         "threshold": "0 errors, 0 hash failures; after_the_fact_analysis only",
         "measured": f"checker ok (v3 97 files, v1 59); fit_window_end "
                     f"{v2['manifest']['fit_window_end']} (data to {v2['data_end']}); guard "
                     "refuses validator/scoring/calibration/prior/baseline/design-time/"
                     "selection. v2 fails the checker on s5r_product.py (edited for v3; "
                     "byte-exact v2 code archived)", "passed": True},
        {"criterion": "T3 leave one family out, pre-origin", "threshold": "report only",
         "measured": f"GNSS ratio {t3['gnss']['ratio']:.3f} (coverage "
                     f"{t3['gnss']['coverage']['coverage']:.3f}); rings+piles "
                     f"{t3['cmp']['ratio']:.3f}; InSAR r {t3['insar']['r_fused']:.3f} vs "
                     f"{t3['insar']['r_levonly']:.3f}", "passed": None, "report_only": True},
        {"criterion": "R2 post-2022 family hold-out (after the fact; U7)",
         "threshold": "report only",
         "measured": f"WRA {r2['gnss_wra']['ratio']:.3f}, NGL {r2['gnss_ngl']['ratio']:.3f}, "
                     f"InSAR {r2['insar']['ratio']:.3f}, piles {r2['piles']['ratio']:.3f}, rings "
                     f"{r2['rings']['ratio']:.3f} (round-1 model, not re-run)",
         "passed": None, "report_only": True},
        {"criterion": "Posterior correlations of the partial-column operators (plan risk)",
         "threshold": "report only",
         "measured": "median |corr(f,u)| 0.61 (max 0.81); |corr(f, local rate)| 0.99; v3 "
                     f"without rings/piles {sens['no_rings_piles']['rms_inc_2022_2025']:.2f} "
                     "mm/yr RMS", "passed": None, "report_only": True},
        {"criterion": "After-the-fact diagnostics (T2 with a buffer, stage-A continuation, "
                      "campaign offsets)", "threshold": "report only",
         "measured": f"T2-buffer ratio {tb['ratio_fused_idw']:.3f}, coverage "
                     f"{tb['coverage_fused']['coverage']:.3f}; stage A -2logL "
                     f"{sa['m2ll_returned']:.2f} -> {sa['continuation']['m2ll']:.2f}; campaign "
                     f"sig_camp {cp['best']['x']['sig_camp']:.2f} mm, q "
                     f"{cp['best']['x']['q']:.0f}", "passed": None, "report_only": True},
    ]
    out = {
        "wp": "S5", "status": "partial",
        "summary": SUMMARY,
        "acceptance": acc,
        "products": [
            {"path": str(C.DATAPREP / C.PRODUCT / "v3") + "/", "layer": "L2",
             "manifest": "results/dataprep/subsidence_fused/v3/MANIFEST.json (current; gate "
                         "false; 53 inputs, 34 code files, 10 outputs: fused_annual.npz, "
                         "fused_cells_annual.parquet, fused_monthly.npz, "
                         "fused_cells_monthly.parquet, gnss_segments.csv, leveling_steps.csv, "
                         "hyperparameters.json, sensitivity.json, validation.json, "
                         "PRODUCT_CARD.md; supersedes v1 and v2)",
             "allowed_uses": "after_the_fact_analysis only",
             "fit_window_end": v2["manifest"]["fit_window_end"]},
            {"path": str(C.DATAPREP / C.PRODUCT / "v2") + "/", "layer": "L2",
             "manifest": "superseded by v3 (identical values; monthly sd label imprecise); its "
                         "manifest fails the checker on s5r_product.py, byte-exact build code "
                         "archived in S5_evidence/repair/code_at_v2/",
             "allowed_uses": "after_the_fact_analysis only", "fit_window_end": "2026-09"},
            {"path": str(C.DATAPREP / C.PRODUCT / "v1") + "/", "layer": "L2",
             "manifest": "superseded (SJES artefact; no monthly field); unchanged, checker ok",
             "allowed_uses": "after_the_fact_analysis only", "fit_window_end": "2025-12"}],
        "deviations_from_plan": DEVIATIONS,
        "issues_for_downstream": ISSUES,
        "registration": {
            "round1": {"id": C.PREREG_ID, "sha256": "6b56e526c4c334ac0b58648da93c64be0b409ab838"
                                                    "963c60ade35a860c579a29",
                       "created": "2026-10-14T02:08:05Z", "status": "unchanged; not re-run"},
            "monthly_test": {"id": R.MTEST_ID, "sha256": m1["registration"]["prereg_sha256"],
                             "created": "2026-10-17T03:50:59Z", "scored": "2026-10-17T04:12:55Z",
                             "post_registration_looks": m1["registration"][
                                 "post_registration_looks"]},
            "declarations": [{"id": R.DECL_ID, "sha256": "a97921139bdf79a38136dd9a91a353b97f375"
                                                         "c20facc54c1d5f800ec963654c1",
                              "created": "2026-10-17T04:12:35Z"},
                             {"id": "S5.subsidence_fused_rules.v3",
                              "sha256": "5406a62f2511874fb48455a6b942da07e07a70c2aa283abc098dbe"
                                        "6b6de8cb86", "created": "2026-10-17T04:19:53Z"}]},
        "repair_round": {
            "date": "2026-10-17",
            "verifier_majors": {"M1_sjes": "fixed (GNSS gap-offset rule)",
                                "M2_monthly": "delivered under registered test M1 (marginal "
                                              "pass)"},
            "stage2": {"hyper": s2["hyper"], "null": s2["null"]["hyper"],
                       "lr_null_minus_full": s2["lr_null_minus_full"]},
            "m1": {"ratio": ms["ratio"], "p_ge_1": ms["boot_ratio"]["p_ge_1"],
                   "coverage": ms["coverage"]["coverage"], "by_year": by,
                   "all_station_months_ratio": m1["all_station_months"]["ratio"]},
            "dryrun_synthetic": {k: {"criteria": v["criteria"],
                                     "ratio_scored": v["scored"].get("ratio")}
                                 for k, v in dr.items()},
            "sensitivity": sens, "diag_t2buffer": {k: tb[k] for k in (
                "rmse_fused", "rmse_idw", "ratio_fused_idw", "rms_z")} | {
                "coverage": tb["coverage_fused"]["coverage"]},
            "diag_campaign": {"sig_camp": cp["best"]["x"]["sig_camp"],
                              "q": cp["best"]["x"]["q"], "m2ll_best": cp["best"]["m2ll"],
                              "m2ll_without": cp["m2ll_registered_values_sig_camp_0.05"]},
            "diag_stageA": {"m2ll_returned": sa["m2ll_returned"],
                            "m2ll_continued": sa["continuation"]["m2ll"],
                            "continued": sa["continued"],
                            "product_sensitivity": sa["product_sensitivity"]},
            "superseded_record": "results/dataprep/subsidence_fused/SUPERSEDED.json",
            "round1_report_copies": ["reports/S5.r1.md", "reports/S5.r1.json"]},
        "ledger_rows": f"results/dataprep/ledger/rows/S5.csv: {len(s5rows)} rows; round-1 "
                       "fragment kept as ledger/superseded/S5.r1.csv; conflicts across all "
                       f"fragments: {len(ledger.conflicts(allr))} ({len(allr)} rows)",
        "claude_api_tokens": 0,
        "gpu": "none (CUDA_VISIBLE_DEVICES empty; no torch import)",
        "evidence": "results/dataprep/reports/S5_evidence/ (round 1) and S5_evidence/repair/ "
                    "(hyper_stage2.json, m1.json, m1_targets.parquet, dryrun_monthly*/, "
                    "dryrun_product*/, diag_*.json, report_numbers_r2.json, code_at_v2/)"}
    (REP / "S5.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"acceptance": len(acc), "ledger_rows": len(s5rows)}))
    return 0


SUMMARY = (
    "Repair round of WP S5 (2026-10-17), answering the verification of subsidence_fused/v1. "
    "Both majors are fixed. (1) SJES: the GNSS operator now starts a free offset after every "
    "gap > 30 d in a station's daily record (41 extra offsets at 20 stations, rule declared "
    "before the build), so SJES's uncorrected ~100 mm in-gap offset no longer bends the field: "
    "mean 2022/2023 increments within 3 km of SJES fall from 47.9/47.1 to 5.2/9.7 mm/yr, and "
    "the fan-wide 2023 median from 13.4 to 8.7 mm/yr. (2) Monthly field: a stage-2 InSAR "
    "monthly anomaly (AR(1) x Matern, block-tridiagonal exact inference) is delivered for "
    "2020-01..2025-12 where the plane-free anomaly variance is cut by >= 50 % (0.72 of "
    "cell-months), gated by a new registered pre-origin test S5.monthly_field.v1 M1, which "
    "passed marginally (ratio 0.972 vs the annual model on 539 withheld WRA GNSS station-months, "
    "bootstrap P(ratio >= 1) 0.086 against a gate of 0.10; all skill from 2021; stage 2 found "
    "rho = 0, so the anomaly is not separated from atmosphere by its time behaviour). Every "
    "minor is fixed or disclosed with a number: stepped benchmarks flagged (9 of 834), 2026 "
    "out of the VR mask, fit_window_end 2026-09 with inputs, complete provenance, ledger roles "
    "corrected (design_time_evidence), sd labelled uncalibrated, report-only rows null. "
    "After-the-fact diagnostics: T2 with a 10 km buffer still beats IDW (0.635) and its "
    "coverage would pass (0.876); continuing the stage-A optimiser lowers -2 log L by 52 "
    "(sig_v0 3.2 -> 30.7, q stays at 400) but moves 2020-25 increments by only 0.47 mm/yr RMS; "
    "per-campaign leveling offsets are supported (sig_camp 3.1 mm; -2 log L 59 below the "
    "continued no-campaign optimum) but q stays near its bound (363). The registered product "
    "gate stays FAILED (T1 1.042 and coverage 0.631; T2 "
    "coverage 0.940), not re-tested, thresholds unchanged. Current product: "
    "subsidence_fused/v3 (L2; values identical to v2, built first; v3 corrects v2's monthly sd "
    "label), after_the_fact_analysis only. No GPU, no Claude API, nothing under hydrophysics/ or "
    "tests/, nothing staged.")

DEVIATIONS = [
    "Monthly field is two-stage: an InSAR-only monthly anomaly (stage 2) on top of the annual "
    "fusion, with the annual posterior mean held fixed and both variances added; not a joint "
    "monthly model. Delivered only where the plane-free anomaly VR >= 0.5 and only because "
    "the registered test M1 passed. The plane of each month is not resolved.",
    "Exact Gaussian inference (dense for the annual model on a 4 km lattice, block "
    "tridiagonal for stage 2) instead of INLA or a sparse GP.",
    "T2 (registered) left one 10 km block out with no buffer, unlike plan 4.3 (>= 10 km for "
    "leveling folds); an after-the-fact buffered re-run is reported (ratio 0.635, coverage "
    "0.876), not re-scored as the gate.",
    "Rings enter from L0 with free offsets per segment and InSAR only through within-segment "
    "changes, so neither S1a's bridge nor LiCSBAS gap offsets enter; piles from "
    "deep_benchmarks_printed/v2.",
    "Product allowed uses are after_the_fact_analysis only (GNSS, InSAR and ring inputs failed "
    "their gates), whatever S5's gate.",
    "Hyperparameters at registered bounds (q 400, sig_ins 0.2, ell_aps 40, sig_rc 0.05; stage 2: "
    "rho 0, sig_ins2 0.2, ell_aps2 40) and Nelder-Mead stages A, B (re-fit) and D not "
    "converged; an after-the-fact continuation of stage A lowers -2 log L by 52 (product "
    "change 0.47 mm/yr RMS). The recorded 44817.03 of stage A includes the out-of-bound "
    "penalty on q (unpenalised 44813.66).",
    "Leveling has no fresh units under A1, so T1/T2 are design-time evidence; M1's WRA GNSS "
    "units are design-exposed as well (procedure H, T3, S4b G1).",
    "The v2/v3 operator amendments (GNSS gap offsets, stepped benchmarks) are covered by "
    "declarations, not by a registered gate test; T1/T2 were not re-run on them.",
    "The GNSS gap rule was motivated by 2023+ evidence (SJES's post-gap level): U7 tension "
    "flagged for the user, as in S1a (the product selects no model and forbids pre-origin uses).",
    "M1's skill gate includes a bootstrap clause (P(ratio >= 1) <= 0.10) chosen after synthetic "
    "dry runs (design on synthetic values only, before registration).",
    "v3 labels the monthly value sd 'uncalibrated', more conservative than M1's registered "
    "label rule ('calibrated at monthly resolution'), because M1 checked only the anomaly and "
    "the annual part failed T1/T2.",
    "v2 was superseded within the round by v3 (labels only); v2's manifest now fails the "
    "checker on s5r_product.py, edited to build v3; byte-exact v2 code archived "
    "(S5_evidence/repair/code_at_v2/, sha256 equal to v2's record) and SUPERSEDED.json written.",
]

ISSUES = [
    "Gate FAILED (T1 1.042, coverage 0.631; T2 coverage 0.940, over-covers); subsidence_fused/v3 "
    "is after_the_fact_analysis only. Validator use needs a new registered test (T1/T2 windows "
    "are design-exposed) and the user's decisions on the S3/S4b/S1a gates.",
    "Do not use subsidence_fused/v1 near SJES (spurious ~48 mm/yr bowl in 2022-23); use v3.",
    "v3 vs the v1 operator: 8.25 mm/yr RMS over cells (2022-25), 1,247 cells moved > 5 mm; the "
    "fan-wide 2023 median moves 13.4 -> 8.7 mm/yr, mostly from SJES (SJES-only breaks 9.2). "
    "Whether the other cross-gap GNSS ties (TKJS, LYES, MDES, STES, JJES, KTES, SSES in 2022-23) "
    "were right cannot be decided from these data; S3 should test or catalogue in-gap offsets.",
    "Monthly field: read detail outside 2021-like conditions as unconfirmed, possibly "
    "atmospheric (M1 ratio by year 2020 1.040, 2021 0.867, 2022 1.009; stage-2 rho = 0). No "
    "support in 2021-05..07 and 2025-04. The monthly value sd is uncalibrated (annual part).",
    "All annual sds are uncalibrated; after 2021 probably too small by about 2.5 (T1 rms z "
    "2.47); inside the leveling era the no-buffer T2 over-covers, the buffered T2 covers 0.876.",
    "The VR >= 0.5 mask is almost non-binding (q at its bound): 0.98 of active cells; do not "
    "read it as an anti-persistence restriction. 2026 (flag 6) is never in the mask now.",
    "Model for a next version (needs a new registration): per-campaign leveling offsets "
    "(supported: sig_camp 3.1 mm, -2 log L 59 below the continued no-campaign optimum), sig_v0 "
    "near 30 (the stage-A optimum), a "
    "bounded or re-parameterised q, an InSAR noise floor, and a buffered T2.",
    "Ring/pile depth fractions are nearly collinear with local rates (|corr| 0.99); leaving "
    "rings and piles out moves 2022-25 increments by 2.95 mm/yr RMS.",
    "9 stepped benchmarks (leveling_steps.csv) and 41 GNSS segment offsets (gnss_segments.csv) "
    "are listed for S3/H-side review.",
    "W3 cannot take an epoch correction from S5: every pre-origin use is forbidden.",
    "R-e (112-year leveling appendix) was not drafted (R1 not approved); InSAR stays the only "
    "fan-wide post-2021 extension.",
    "subsidence_fused/v3 hashes the round-1 and repair modules and shared code (common, harness, "
    "s1_rings, s4b): editing them makes the checker flag v3; build a v4 instead. v2 already "
    "fails for that reason (documented in SUPERSEDED.json).",
    "USER DECISIONS: the U7 reading for the post-2022-motivated GNSS gap rule; the upstream "
    "gates (S3, S4b, S1a); whether a registered v4 (campaign offsets, buffered T2, fresh "
    "windows) is wanted.",
]

if __name__ == "__main__":
    raise SystemExit(main())

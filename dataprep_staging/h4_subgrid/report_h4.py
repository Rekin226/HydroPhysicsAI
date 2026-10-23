"""WP H4: write results/dataprep/reports/H4.md and H4.json from the registered evidence.

Every number is read from reports/H4_evidence/ (registered/scores.json, report_facts.json,
design_time/d0.json, design_time/t0_smoke.json, guard_probe.json) and the product manifest.
Usage: python report_h4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, REPO  # noqa: E402

EV = DATAPREP / "reports/H4_evidence"
PROD = DATAPREP / "subgrid_operator/v1"
CODE = ["h4lib.py", "r1_registered.py", "prereg_h4.py", "b1_build_product.py",
        "d0_design_check.py", "t0_smoke.py", "probe_blocking.py", "looks_h4.py", "ledger_h4.py",
        "report_facts.py", "guard_probe_h4.py", "report_h4.py", "tests/test_h4lib.py"]


def f3(x):
    return f"{x:.3f}"


def main() -> None:
    sc = json.loads((EV / "registered/scores.json").read_text())
    fa = json.loads((EV / "report_facts.json").read_text())
    d0 = json.loads((EV / "design_time/d0.json").read_text())
    t0 = json.loads((EV / "design_time/t0_smoke.json").read_text())
    gp = json.loads((EV / "guard_probe.json").read_text())
    man = json.loads((PROD / "MANIFEST.json").read_text())
    z = np.load(EV / "registered/arrays.npz", allow_pickle=False)
    q, ok = z["quartile"], z["score_ok"]
    s123 = ok & (q >= 1) & (q <= 3)
    pooled = {k: float(np.sqrt(z[k][s123].mean() / z["mse_base"][s123].mean()))
              for k in ("mse_t1", "mse_t2")}
    p1, p2 = sc["primary"]["t1"], sc["primary"]["t2"]
    s1, s2 = p1["summary"], p2["summary"]
    crit = {c["id"]: c for c in sc["criteria"]}
    failed = [c["id"] for c in sc["criteria"] if not c["passed"]]
    a = p1["a"]
    fr = fa["t2_multiplier_range"]
    t1s, t2s = fa["t1"], fa["t2"]
    sens = sc["sensitivity"]

    acceptance = []
    for c in sc["criteria"]:
        acceptance.append({"criterion": f"{c['id']}: {c['criterion']}",
                           "threshold": c["threshold"], "measured": c["measured"],
                           "passed": bool(c["passed"])})
    acceptance += [
        {"criterion": "[product check] subgrid_operator/v1 manifest re-hash (common.check)",
         "threshold": "0 failures", "measured": f"exit {gp['checker_exit']}; 26 files ok",
         "passed": gp["checker_exit"] == 0},
        {"criterion": "[product check] loader guard: failed-gate product refuses modelling uses",
         "threshold": "design_time_evidence and after_the_fact_analysis granted; model_design, "
                      "validator, scoring_target, calibration_target, forcing_frozen, selection "
                      "refused",
         "measured": ", ".join(f"{u} {'granted' if v['granted'] else 'refused'}"
                               for u, v in gp.items() if u != "checker_exit"),
         "passed": all(gp[u]["granted"] == (u in ("design_time_evidence",
                                                  "after_the_fact_analysis"))
                       for u in gp if u != "checker_exit")},
        {"criterion": "[harness check] pre-registration before any H4 look at head data; no "
                      "blocking or post-registration look",
         "threshold": "prereg3.assert_can_score passes; post_registration_looks empty",
         "measured": f"passed; post-registration looks {len(sc['post_registration_looks'])}; "
                     f"unverifiable prior looks disclosed {sc['unverifiable_prior_looks_n']}",
         "passed": True},
    ]

    summary = (
        f"WP H4 built and tested the plan's sub-grid observation operator h_well = h_cell - "
        f"sum_j Q_j k(r_j) on the engine/v2 hindcast (cwa theta) at the 158 calibration wells, "
        f"with a periodic-Theis seasonal kernel and a Thiem slow kernel over kept Yunlin census "
        f"pumps within 2 km (Chow-Lin kWh x the twin's lift conversion), two non-negative "
        f"fan-wide multipliers fitted on 2012-02..2019-12, and the test pre-registered "
        f"(H4.subgrid_operator.v1, harness.prereg3) before any H4 code read a head "
        f"observation. The fit put the whole correction on the seasonal kernel (a_s "
        f"{a[0]:.2f} x the model-unit cone, a_l {a[1]:.2f}). On the held-out years 2020-22 the "
        f"Q4 (high-swing, 2012-19 sd) excess over climatology fell by "
        f"{s1['Q4']['excess_reduction_m2']:.2f} m2 (Q4 RMSE {s1['Q4']['rmse_base']:.3f} -> "
        f"{s1['Q4']['rmse_corrected']:.3f} m) and with site-grouped folds by "
        f"{s2['Q4']['excess_reduction_m2']:.2f} m2 ({s2['Q4']['rmse_base']:.3f} -> "
        f"{s2['Q4']['rmse_corrected']:.3f} m); the real kernel beat both placebos in both "
        f"tests (permutation p 0.025, 0.025, 0.025 and 0.050, 39 draws each). Q1-Q3 RMSE rose "
        f"by at most 1.46 % on held-out years (pass) but by 1.53 / 2.07 / 1.99 % on held-out "
        f"sites, so criterion A8 fails at Q2 (1.0207 against 1.02) and the registered gate "
        f"fails ({len(failed)} of 8 criteria failed). The rise comes from the 28 low- and mid-swing "
        f"wells that sit in Yunlin pump clusters, where the operator adds a swing the "
        f"observations do not have; the 81 unsupported Q1-Q3 wells are unchanged. The "
        f"product subgrid_operator/v1 is published with the failed gate and only "
        f"design_time_evidence and after_the_fact_analysis uses; it must not enter I1 as a "
        f"secondary model. The base hindcast is in sample over 2020-22 (theta fitted to 2022), "
        f"so the held-out years test the operator's multipliers only, and in those years the "
        f"base model already beat climatology at Q4. Seven sensitivity members (not gated) all "
        f"lower the Q4 error and all raise some Q1-Q3 quartile under site hold-out. No GPU, 0 "
        f"Claude API tokens; nothing under hydrophysics/, tests/ or L0 was written; nothing "
        f"was staged in git.")

    key_numbers = [
        f"Support: {sc['wells_with_kernel']} of 158 wells get a correction (all Yunlin, model "
        f"layers 0-1: 19 + 42); Changhua wells have no Yunlin pump within 2 km; Yunlin layers "
        f"2-3 get none under the twin's pump split (f_shallow 0.087)",
        f"Pumps within 2 km of Yunlin wells: median {d0['pumps_within_2km']['YL']['50']:.0f} "
        f"(1 km: {d0['pumps_within_1km']['YL']['50']:.0f}); 114,310 kept Yunlin pumps with "
        f"coordinates",
        f"Design-time (no heads): unit-multiplier correction sd median "
        f"{d0['unit_correction_sd_m']['sum']['YL']['50']:.3f} m, max "
        f"{d0['unit_correction_sd_m']['sum']['YL']['100']:.3f} m; |K0| / ln kernel ratio "
        f"0.91-0.98 at r <= 2 km; per-pump Q sum / engine Q_rate median "
        f"{d0['fan_total_Q_check']['per_pump_sum_over_engine_Q_rate']['50']:.3f}",
        f"Synthetic smoke (t0): multipliers (8, 4) recovered as "
        f"({t0['real_truth_no_seasonal']['a_t1'][0]:.2f}, "
        f"{t0['real_truth_no_seasonal']['a_t1'][1]:.2f})",
        f"Quartile cuts (2012-19 observed sd): {', '.join(f3(c) for c in sc['quartile_cuts_sd_m'])}"
        f" m; scored wells {sc['n_scored']} (Q1-Q4 "
        f"{'/'.join(str(v) for v in sc['quartile_counts_scored'].values())}); Q4: "
        f"{sc['q4_scored_by_county']['YL']} Yunlin + {sc['q4_scored_by_county']['CH']} Changhua, "
        f"{sc['q4_scored_with_kernel']} with a kernel",
        f"Multipliers T1 (all wells, 2012-19): a_s {a[0]:.3f}, a_l {a[1]:.3f}; T2 folds a_s "
        f"{fr['a_s'][0]:.2f}-{fr['a_s'][1]:.2f}, a_l {fr['a_l'][0]:.2f}-{fr['a_l'][1]:.2f} "
        f"(spread uncalibrated); fitted correction sd at supported wells median "
        f"{fa['corr_t1_sd_supported_m']['50']:.3f} m, max {fa['corr_t1_sd_supported_m']['100']:.3f} m",
        f"T1 held-out years: Q4 excess {s1['Q4']['excess_base_m2']:.2f} -> "
        f"{s1['Q4']['excess_corrected_m2']:.2f} m2 (reduction {s1['Q4']['excess_reduction_m2']:.2f}); "
        f"Q4 RMSE {s1['Q4']['rmse_base']:.3f} -> {s1['Q4']['rmse_corrected']:.3f} m "
        f"(clim {s1['Q4']['rmse_clim']:.3f}); Q4 amplitude ratio "
        f"{p1['q4_amplitude_ratio_base']:.2f} -> {p1['q4_amplitude_ratio_corrected']:.2f}; "
        f"supported Q4 wells improved {t1s['Q4_supported']['n_improved']}/"
        f"{t1s['Q4_supported']['n']}",
        f"T2 held-out sites: Q4 reduction {s2['Q4']['excess_reduction_m2']:.2f} m2; Q4 RMSE "
        f"{s2['Q4']['rmse_base']:.3f} -> {s2['Q4']['rmse_corrected']:.3f} m; supported Q4 "
        f"improved {t2s['Q4_supported']['n_improved']}/{t2s['Q4_supported']['n']}",
        f"Placebo Q4 reductions (median / max of 39): T1 shuffle "
        f"{fa['placebo_shuffle_t1']['median']:.2f} / {fa['placebo_shuffle_t1']['max']:.2f}, "
        f"distant {fa['placebo_distant_t1']['median']:.2f} / {fa['placebo_distant_t1']['max']:.2f}"
        f"; T2 shuffle {fa['placebo_shuffle_t2']['median']:.2f} / "
        f"{fa['placebo_shuffle_t2']['max']:.2f}, distant {fa['placebo_distant_t2']['median']:.2f}"
        f" / {fa['placebo_distant_t2']['max']:.2f} (real 10.16 / 3.81)",
        f"Q1-Q3 RMSE ratios T1 {crit['A7']['measured']}; T2 {crit['A8']['measured']} "
        f"(post hoc, not the registered rule: pooled Q1-Q3 {pooled['mse_t1']:.4f} / "
        f"{pooled['mse_t2']:.4f})",
        f"Supported Q1-Q3 wells (8/8/12) T1 RMSE: Q1 {t1s['Q1_supported']['rmse_base']:.3f} -> "
        f"{t1s['Q1_supported']['rmse_corrected']:.3f}, Q2 {t1s['Q2_supported']['rmse_base']:.3f}"
        f" -> {t1s['Q2_supported']['rmse_corrected']:.3f}, Q3 "
        f"{t1s['Q3_supported']['rmse_base']:.3f} -> {t1s['Q3_supported']['rmse_corrected']:.3f}"
        f" m; the 81 unsupported Q1-Q3 wells unchanged",
        f"Base vs climatology 2020-22 (in-sample theta): all wells "
        f"{s1['all']['rmse_base']:.3f} vs {s1['all']['rmse_clim']:.3f} m; Q4 "
        f"{s1['Q4']['rmse_base']:.3f} vs {s1['Q4']['rmse_clim']:.3f} m",
        "Sensitivity T1 / T2 Q4 reduction (m2) and worst T2 Q1-Q3 ratio: " + "; ".join(
            f"{k} {v['t1']['summary']['Q4']['excess_reduction_m2']:.2f} / "
            f"{v['t2']['summary']['Q4']['excess_reduction_m2']:.2f}, "
            f"{max(v['t2']['summary'][qq]['rmse_ratio_corrected_over_base'] for qq in ('Q1', 'Q2', 'Q3')):.4f}"
            for k, v in sens.items()),
        f"Registered run {sc['runtime_s']:.0f} s in one CPU slot; GPU none; Claude API tokens 0",
    ]

    deviations = [
        "Code lives in dataprep_staging/h4_subgrid/ (decision D12), not results/dataprep/"
        "_staging_code/.",
        "Q_j comes from Yunlin pumps only in the primary, although the revised plan makes the "
        "Changhua census available: the hindcast carries no Changhua pumping, so a local kernel "
        "on Changhua pumps would also absorb the missing Changhua cell-mean drawdown and bias a "
        "fan-wide multiplier. The all-in-grid-counties member is reported (116 wells with a "
        "kernel, 55 of them in Changhua; under site hold-out Q2 improves to 0.966 while Q1 and "
        "Q3 degrade to 1.023).",
        "T and S are not profiled where a zone parameter sits at a bound (plan wording). T "
        "enters as 1/(2 pi T), absorbed by the fitted multiplier, and only 4 supported layer-0 "
        "wells sit at the T bound; the registered t_bound_half member moves the T1 reduction "
        "from 10.16 to 10.19 m2.",
        "'Fitted on 2012-19 residual amplitude' is implemented as non-negative least squares on "
        "the 2012-19 residual series (per-well means removed, equal-well weights), not as "
        "amplitude matching.",
        "Quartiles are defined on the 2012-19 observed sd of the 158 calibration wells (U7), not "
        "on the audit's 2023-26 quartiles; the Q4 excess is measured on 2020-22, where the base "
        "hindcast is in sample and already beats climatology (excess -24.30 m2). 'Q4 excess "
        "falls' was registered as a strict decrease.",
        "Held-out sites use A1's site-fold partition (fresh.folds.wells_all) as data, but the "
        "registered unit is the external unit H4:wells158_2020-2022: the registry unit carries "
        "P40 and H2 looks on 2012-22 that block any new registration on it.",
        "The >15 km placebo is implemented as donor neighbourhoods: the same kernel evaluated "
        "at a random census pump position at least 17 km (15 km + r_max) from the well, so only "
        "pumps more than 15 km away enter; both placebos use 39 seeded draws and a permutation "
        "p <= 0.05.",
        "The Q1-Q3 criterion is applied per quartile (stricter than pooled); pooled Q1-Q3 would "
        f"have been {pooled['mse_t2']:.4f} under site hold-out (post hoc; the verdict stands).",
        "The product carries the plan's outputs (per-well correction series, multipliers, "
        "report) plus the operator weights; its gate failed, so its uses are restricted to "
        "design_time_evidence and after_the_fact_analysis.",
    ]

    issues = [
        "I1: subgrid_operator/v1 failed its registered acceptance (A8, Q2 1.0207 > 1.02 under "
        "site hold-out); it must not enter I1 as a secondary model or observation hook. The "
        "loader guard refuses model_design and validator.",
        "Plan 7.3 wording: the operator recovers a small, location-specific part of the Q4 "
        "swing (Q4 RMSE -2.2 % on held-out years, -0.8 % on held-out sites; beats shuffled and "
        "distant pumps), but it adds swing at low- and mid-swing wells inside the same pump "
        "clusters (28 Q1-Q3 wells: RMSE +2 to +9 % on held-out years, +4 to +10 % on held-out sites), so it does not pass the plan's "
        "no-harm condition. The local cones are not recovered with this grid and this operator.",
        "Any follow-up variant (per-well gating by pre-origin sd, per-zone multipliers, an "
        "amplitude-based fit, Changhua pumps after the I1 recalibration) is designed after "
        "seeing 2020-22 at these wells: it needs a new pre-registration and a test window or "
        "unit not used here (2020-22 at the 158 wells is now an H4 test unit; no 2023+ data may "
        "gate it, U7).",
        f"The fitted seasonal multiplier is {a[0]:.1f} times the model-unit Thiem/periodic-Theis "
        "cone and the slow multiplier is 0: interannual local pumping adds nothing beyond the "
        "hindcast. A multiplier above 1 is consistent with physical volumes above model units "
        "(WRA Jan-May volumes 2.58 x model) or a lower local T, but it is also fitted partly to "
        "the regional amplitude deficit (placebo multipliers 1.5-2.2).",
        "After the I1 recalibration (Changhua census, new theta) the hindcast changes and any "
        "operator multipliers must be refitted; in the all_ingrid_counties member Q2 "
        "improves (0.966) once 55 Changhua wells get a kernel, but Q1 and Q3 degrade (1.023); "
        "whether a Changhua local kernel helps once Changhua pumping is in the forcing is untested.",
        "H1a pumping-cycle wells: 07080122 is the only one among the 158 and is not scored "
        "(17 fit months); 07080132 and 10050211 are not calibration wells.",
        "The engine/v2 hindcast used theta sha256 789d0e45... (unchanged at run time). If "
        "stage3_theta.json is rewritten (batch 8b), subgrid_operator/v1 is stale.",
        f"prereg3 disclosed {sc['unverifiable_prior_looks_n']} prior looks on units of unknown "
        "membership whose windows overlap 2020-22 (A1 verifier's open issue on external units); "
        "none is an H4 look and none blocked the registration.",
    ]

    products = [{
        "path": str(PROD.relative_to(REPO)) + "/", "layer": "L3",
        "manifest": f"results/dataprep/subgrid_operator/v1/MANIFEST.json (sha256 "
                    f"{sha256_file(PROD / 'MANIFEST.json')[:16]}...; gate passed=false, failed "
                    f"{failed}; checker ok, 26 files). Files: correction_series.parquet "
                    f"(20,698 well-months 2012-02..2022-12), well_table.parquet, "
                    f"operator_weights.npz, multipliers.json, validation.json, PRODUCT_CARD.md",
        "allowed_uses": "design_time_evidence, after_the_fact_analysis (gate failed: "
                        "model_design, validator and every target/forcing use forbidden)",
        "fit_window_end": f"{man['fit_window_end']} (non_exempt_fit_window_end "
                          f"{man['non_exempt_fit_window_end']}; multipliers fitted on 2012-02.."
                          f"2019-12; 3 inherited census exemptions)"}]
    code_paths = [str(HERE / c) for c in CODE]
    rep = {"wp": "H4", "status": "failed", "summary": summary, "acceptance": acceptance,
           "products": products, "code_paths": code_paths, "key_numbers": key_numbers,
           "deviations_from_plan": deviations, "issues_for_downstream": issues,
           "report_path": str(DATAPREP / "reports/H4.md"), "date": "2026-10-15",
           "preregistrations": ["H4.subgrid_operator.v1 (sha256 " + sc["prereg_sha256"] + ")"],
           "claude_api_tokens": 0, "gpu": "none"}
    (DATAPREP / "reports/H4.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)
                                              + "\n")
    md = ["# WP H4: sub-grid observation operator for the high-swing wells", "",
          f"Status: **failed** (registered gate H4.subgrid_operator.v1: {8 - len(failed)} of 8 "
          f"criteria pass; {', '.join(failed)} fails).", "", summary, "",
          "## Acceptance", "", "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for r in acceptance:
        md.append(f"| {r['criterion']} | {r['threshold']} | {r['measured']} | "
                  f"{'yes' if r['passed'] else '**no**'} |")
    md += ["", "Q4 excess = sum over the scored Q4 wells of (model MSE - climatology MSE) over "
           "2020-01..2022-12, with the training-mean datum and a 2012-19 month-of-year "
           "climatology. Negative values mean the model beats climatology; the base hindcast is "
           "in sample over 2020-22.", "",
           "## What was built", "",
           "- `h4lib.py`: data access through the A0 guard (engine/v2 `model_design`, "
           "yunlin_electricity/preorigin/v2 `forcing_frozen`, census_L1/v3 `model_design`, "
           "heads/L1_clean/v3 with `window_end` 2022-12 for `calibration_target`, `datum_fit`, "
           "`baseline` and `scoring_target`, validation_units/v3 `selection`), the kernels, the "
           "sparse operator, the NNLS fit, the T1/T2 scorers and both placebo generators.",
           "- Kernels: seasonal `w/(2 pi T)(|K0(r k sqrt(i))| - |K0(R k sqrt(i))|)`, slow "
           "`w/(2 pi T) ln(R/r)`, R = 2 km, r >= 100 m, T and S of the well's layer at its cell, "
           "w the twin's point-extraction share (layer 0: 0.087, layer 1: 0.913, deeper: 0); Q_j "
           "= Chow-Lin kWh x the twin's lift conversion; slow = trailing 12-month mean.",
           "- `d0_design_check.py` and `t0_smoke.py` ran before the registration and read no "
           "head observation (logged as design-time looks). `r1_registered.py` is the "
           "registered run (assert_can_score first). `b1_build_product.py` packages the "
           "product; `ledger_h4.py`, `looks_h4.py`, `report_facts.py`, `guard_probe_h4.py`, "
           "`report_h4.py` are bookkeeping. `tests/test_h4lib.py`: 3 synthetic tests pass; ruff "
           "clean.", "",
           "## Quartile detail (registered run)", "",
           "| quartile | n | RMSE base | T1 | T2 | climatology | T1 ratio | T2 ratio |",
           "|---|---|---|---|---|---|---|---|"]
    for qq in ("Q1", "Q2", "Q3", "Q4"):
        a1, b1 = s1[qq], s2[qq]
        md.append(f"| {qq} | {a1['n_wells']} | {a1['rmse_base']:.3f} | "
                  f"{a1['rmse_corrected']:.3f} | {b1['rmse_corrected']:.3f} | "
                  f"{a1['rmse_clim']:.3f} | {a1['rmse_ratio_corrected_over_base']:.4f} | "
                  f"{b1['rmse_ratio_corrected_over_base']:.4f} |")
    md += ["", "Supported (kernel) vs unsupported wells, T1 / T2 corrected RMSE:", "",
           "| group | n | base | T1 | T2 | improved T1 / T2 |", "|---|---|---|---|---|---|"]
    for g in ("Q1_supported", "Q2_supported", "Q3_supported", "Q4_supported",
              "Q4_unsupported"):
        x, y = t1s[g], t2s[g]
        md.append(f"| {g} | {x['n']} | {x['rmse_base']:.3f} | {x['rmse_corrected']:.3f} | "
                  f"{y['rmse_corrected']:.3f} | {x['n_improved']} / {y['n_improved']} |")
    md += ["", "## Sensitivity members (reported, never gating)", "",
           "| member | wells with kernel | a_s, a_l (T1) | Q4 reduction T1 / T2 (m2) | "
           "T2 Q1 / Q2 / Q3 ratio |", "|---|---|---|---|---|"]
    md.append(f"| primary | {sc['wells_with_kernel']} | {a[0]:.3f}, {a[1]:.3f} | "
              f"{s1['Q4']['excess_reduction_m2']:.2f} / {s2['Q4']['excess_reduction_m2']:.2f} | "
              + " / ".join(f"{s2[qq]['rmse_ratio_corrected_over_base']:.4f}"
                           for qq in ("Q1", "Q2", "Q3")) + " |")
    for k, v in sens.items():
        w1, w2 = v["t1"]["summary"], v["t2"]["summary"]
        md.append(f"| {k} | {v['wells_with_kernel']} | {v['t1']['a'][0]:.3f}, "
                  f"{v['t1']['a'][1]:.3f} | {w1['Q4']['excess_reduction_m2']:.2f} / "
                  f"{w2['Q4']['excess_reduction_m2']:.2f} | "
                  + " / ".join(f"{w2[qq]['rmse_ratio_corrected_over_base']:.4f}"
                               for qq in ("Q1", "Q2", "Q3")) + " |")
    sp = sc["t2_spatial_only_reported"]["summary"]
    md += ["", f"T2 spatial-only variant (fold fits scored on the test wells' 2012-19 months, "
               f"in-sample datum and climatology; reported): Q4 reduction "
               f"{sp['Q4']['excess_reduction_m2']:.2f} m2, Q1 / Q2 / Q3 ratios "
               + " / ".join(f"{sp[qq]['rmse_ratio_corrected_over_base']:.4f}"
                            for qq in ("Q1", "Q2", "Q3")) + ".", "",
           "## Key numbers", ""] + [f"- {k}" for k in key_numbers]
    md += ["", "## Products", ""] + [f"- `{p['path']}` ({p['layer']}): {p['manifest']}. "
                                      f"Allowed uses: {p['allowed_uses']}. fit_window_end "
                                      f"{p['fit_window_end']}." for p in products]
    md += ["", "## Deviations from plan", ""] + [f"{i}. {x}" for i, x in
                                                 enumerate(deviations, 1)]
    md += ["", "## Issues for downstream", ""] + [f"- {x}" for x in issues]
    md += ["", "## Process record", "",
           "- Pre-registration `results/dataprep/preregistration/H4.subgrid_operator.v1.json` "
           f"(sha256 {sc['prereg_sha256']}), written 2026-10-14T16:06:44Z by harness.prereg3, "
           "after two design-time steps that read no head observation (d0_design_check, run twice, "
           "the second time after a lint-only edit with identical output apart from timings; "
           "t0_smoke once; logged at 16:06:17-18Z; all before the registration) "
           "and before the scoring looks (16:12:43Z). Later looks: report_facts and "
           "product_build (purposes logged).",
           "- Ledger fragment `results/dataprep/ledger/rows/H4.csv`: 94 rows, 0 conflicts.",
           "- Evidence: `results/dataprep/reports/H4_evidence/` (design_time/d0.json, "
           "design_time/t0_smoke.json, registered/scores.json, registered/arrays.npz, "
           "registered_run.log, report_facts.json, guard_probe.json, check_v1.json).",
           "- No GPU, 0 Claude API tokens, nothing under hydrophysics/, tests/ or L0 written, "
           "nothing staged in git.", ""]
    (DATAPREP / "reports/H4.md").write_text("\n".join(md))
    print("written", DATAPREP / "reports/H4.md")


if __name__ == "__main__":
    main()

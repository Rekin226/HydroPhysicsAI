"""Write the W1 report (results/dataprep/reports/W1.md, W1.json) and its extra ledger rows.

    python -m w1_canal.report            (from dataprep_staging/, after publish and verify)

Every number is read from the published products' validation.json and from
reports/W1_evidence/verify.json; the narrative is fixed text written after the results
were known (it says so where it interprets them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal.prereg import PREREG_ID  # noqa: E402
from w1_canal.verify import EVIDENCE, latest  # noqa: E402

REPORTS = DATAPREP / "reports"
CODE = ["__init__.py", "config.py", "parse_yearbook.py", "sources.py", "composite.py",
        "diversion.py", "validate.py", "statespace.py", "grid.py", "checks.py", "build.py",
        "publish.py", "prereg.py", "looks.py", "verify.py", "report.py"]


def f3(x):
    return f"{x:.3f}"


def load():
    pre, full = latest("preorigin"), latest("full")
    vp = json.loads((pre / "validation.json").read_text())
    vf = json.loads((full / "validation.json").read_text())
    ver = json.loads((EVIDENCE / "verify.json").read_text())
    return pre, full, vp, vf, ver


def extra_ledger_rows(pv: dict) -> list[dict]:
    yb = "data_fetch_api/new_measurements/wra_hydro_yearbook"
    jj = rel_to_repo(C.JIJI_CSV)
    v3 = rel_to_repo(C.V3_NPZ)
    looks = "results/dataprep/preregistration/looks.jsonl"
    rows = []

    def add(n, s, r, v, note=""):
        rows.append({"number_id": n, "source": s, "role": r, "product_version": v,
                     "note": note})

    for v, ver in pv.items():
        add(f"W1.{v}.denton_exact", jj, "input", ver, "annual district totals")
        add(f"W1.{v}.denton_exact", yb, "input", ver)
        add(f"W1.{v}.indicator_annual_fit", jj, "fit_target", ver)
        add(f"W1.{v}.indicator_annual_fit", yb, "input", ver)
        add(f"W1.{v}.composite_donor_swaps", yb, "input", ver)
        add(f"W1.{v}.composite_donor_swaps", jj, "input", ver, "weir annual inflow")
        add(f"W1.{v}.k_v3level", v3, "input", ver)
        add(f"W1.{v}.k_v3level", jj, "input", ver)
        add(f"W1.{v}.infan_fraction", v3, "input", ver, "LUIMAP 2024 farmland (exemption)")
        add(f"W1.{v}.statespace_hyper", jj, "input", ver)
        add(f"W1.{v}.statespace_hyper", rel_to_repo(C.BRIEFING_110_PDF), "input", ver,
            "Jan-May totals as observed aggregates (product run)")
        add(f"W1.{v}.statespace_hyper", rel_to_repo(C.MOA_691_CSV), "input", ver)
        add(f"W1.{v}.statespace_hyper", yb, "input", ver)
    full = pv["full"]
    add("W1.allocation.v3_rebuild", v3, "validator", full, "v3 field rebuilt from its own "
        "district volumes with the recovered allocation")
    add("W1.parse.coverage", yb, "input", full, "station-years, missing days, pages")
    for ver in pv.values():
        add("W1.check.manifests", f"results/dataprep/{ver}/MANIFEST.json", "input", ver)
        add("W1.check.guard_probes", f"results/dataprep/{ver}/MANIFEST.json", "input", ver)
    add("W1.looks", looks, "input", pv["preorigin"], "seed (scout), late-logged dry run, "
        "product build")
    return rows


CITED = (["W1.preregistration", "W1.looks", "W1.parse.printed_monthly_off", "W1.parse.coverage",
          "W1.parse.days_vs_pilot", "W1.check.jiji_csv_vs_html", "W1.check.janmay_vs_pilot",
          "W1.allocation.v3_rebuild", "W1.check.manifests", "W1.check.guard_probes",
          "W1.preorigin.head_pcorr", "W1.preorigin.luimap_sensitivity"]
         + [f"W1.{v}.{k}" for v in ("preorigin", "full") for k in (
             "beta_a", "shares_final", "statespace_coverage", "cropshare_vs_moa691",
             "composite_scale", "annual_infan_volume", "denton_exact", "indicator_annual_fit",
             "composite_donor_swaps", "k_v3level", "infan_fraction", "statespace_hyper")]
         + [f"W1.{v}.janmay_rmse.{m}.{d}" for v in ("preorigin", "full") for d in ("chc", "yl")
            for m in ("supply_limited_denton", "calendar_denton",
                      "supply_limited_raw_indicator", "clim_fraction_loo", "v3")]
         + [f"W1.{v}.janmay_boot.vs_{b}.{d}" for v in ("preorigin", "full")
            for b in ("clim_fraction_loo", "v3") for d in ("chc", "yl")])


def bench_table(b: dict) -> list[str]:
    s = b["scores"]
    out = ["| method | RMSE Changhua | bias Changhua | RMSE Yunlin | bias Yunlin |",
           "|---|---|---|---|---|"]
    for m in ("supply_limited_denton", "supply_limited_raw_indicator", "calendar_denton",
              "clim_fraction_loo", "v3"):
        out.append(f"| {m} | {f3(s[m]['chc']['rmse'])} | {s[m]['chc']['bias']:+.3f} | "
                   f"{f3(s[m]['yl']['rmse'])} | {s[m]['yl']['bias']:+.3f} |")
    return out


def compare_table(b: dict) -> list[str]:
    out = ["| comparison | district | RMSE difference | 90 % paired bootstrap | sign test "
           "(wins/n, two-sided p) |", "|---|---|---|---|---|"]
    for base in ("clim_fraction_loo", "v3"):
        for d in ("chc", "yl"):
            r = b["compare"][base][d]
            out.append(f"| v4 - {base} | {d} | {r['rmse_diff']:+.3f} | [{r['boot_lo']:+.3f}, "
                       f"{r['boot_hi']:+.3f}] | {r['sign_wins']}/{r['sign_n']}, "
                       f"p = {r['sign_p_two_sided']:.3f} |")
    return out


def main():
    pre, full, vp, vf, ver = load()
    pv = {"preorigin": f"{C.PRODUCT_BASE}/preorigin/{pre.name}",
          "full": f"{C.PRODUCT_BASE}/full/{full.name}"}
    ledger.append_rows(C.WP, extra_ledger_rows(pv))
    missing = ledger.unledgered(CITED)
    conflicts = ledger.conflicts(ledger.read_all())
    P, F = vp["variant_validation"], vf["variant_validation"]
    com = vp["common_checks"]
    bp, bf = P["benchmark"], F["benchmark"]
    ap, af = bp["acceptance"], bf["acceptance"]
    hc = com["head_check"]
    cvp, cvf = P["state_space_loo_coverage"], F["state_space_loo_coverage"]
    pq = com["parse_qc"]
    pvp = com["parse_vs_pilot"]
    sens = P["allocation"]["luimap_sensitivity"]
    ex = ver["exactness"]

    def boot(b, base, d):
        r = b["compare"][base][d]
        return f"{r['rmse_diff']:+.3f} [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}]"

    acceptance = [
        {"criterion": "PRIMARY (pre-registered; pre-origin variant): v4 beats the "
                      "climatological fraction, Changhua, with the 90 % paired-bootstrap "
                      "interval of the RMSE difference below zero",
         "threshold": "RMSE diff < 0 and bootstrap upper bound < 0",
         "measured": boot(bp, "clim_fraction_loo", "chc"),
         "passed": ap["interval_excludes_zero"]["clim_fraction_loo"]["chc"]},
        {"criterion": "PRIMARY: v4 beats the climatological fraction, Yunlin",
         "threshold": "RMSE diff < 0 and bootstrap upper bound < 0",
         "measured": boot(bp, "clim_fraction_loo", "yl"),
         "passed": ap["interval_excludes_zero"]["clim_fraction_loo"]["yl"]},
        {"criterion": "PRIMARY: v4 beats v3, Changhua",
         "threshold": "RMSE diff < 0 and bootstrap upper bound < 0",
         "measured": boot(bp, "v3", "chc"),
         "passed": ap["interval_excludes_zero"]["v3"]["chc"]},
        {"criterion": "PRIMARY: v4 beats v3, Yunlin",
         "threshold": "RMSE diff < 0 and bootstrap upper bound < 0",
         "measured": boot(bp, "v3", "yl"),
         "passed": ap["interval_excludes_zero"]["v3"]["yl"]},
        {"criterion": "PRIMARY overall (all four comparisons); pre-origin variant",
         "threshold": "4/4",
         "measured": f"{sum(v for b in ap['interval_excludes_zero'].values() for v in b.values())}"
                     f"/4; Jan-May LOO RMSE {f3(bp['scores']['supply_limited_denton']['chc']['rmse'])}"
                     f" / {f3(bp['scores']['supply_limited_denton']['yl']['rmse'])} (Changhua / "
                     f"Yunlin, 10^8 m3) vs climatology "
                     f"{f3(bp['scores']['clim_fraction_loo']['chc']['rmse'])} / "
                     f"{f3(bp['scores']['clim_fraction_loo']['yl']['rmse'])} and v3 "
                     f"{f3(bp['scores']['v3']['chc']['rmse'])} / "
                     f"{f3(bp['scores']['v3']['yl']['rmse'])}",
         "passed": ap["passed"]},
        {"criterion": "SECONDARY (reported, same rule): full variant 2012-2025",
         "threshold": "4/4",
         "measured": f"{sum(v for b in af['interval_excludes_zero'].values() for v in b.values())}"
                     f"/4; clim yl {boot(bf, 'clim_fraction_loo', 'yl')}, v3 chc "
                     f"{boot(bf, 'v3', 'chc')}; label '{af['label']}'",
         "passed": af["passed"]},
        {"criterion": "Supporting (pre-registered 'not worse'): head-anomaly partial "
                      "correlation given 3-month rain, v4 pre-origin >= v3 in each of 4 groups "
                      "(Changhua/Yunlin x layer 1/2, 2012-2022)",
         "threshold": "v4 - v3 >= 0 in 4/4 groups",
         "measured": ", ".join(f"{g} {hc[g]['v4_minus_v3_partial']:+.4f}" for g in hc),
         "passed": bool(com["head_check_not_worse"])},
        {"criterion": "Supporting (pre-registered): state-space leave-one-year-out 90 % "
                      "interval coverage of the held-out Jan-May totals, pre-origin "
                      "(Clopper-Pearson 95 % at n_eff = 7 must contain 0.90)",
         "threshold": "interval contains 0.90",
         "measured": f"{cvp['hits']}/{cvp['n']} = {cvp['coverage']:.2f}; CP "
                     f"[{cvp['clopper_pearson_95_neff'][0]:.3f}, "
                     f"{cvp['clopper_pearson_95_neff'][1]:.3f}]; A1 gate "
                     f"[{cvp['a1_harness_gate']['lo']:.3f}, {cvp['a1_harness_gate']['hi']:.3f}]",
         "passed": bool(cvp["passed_prereg_rule"])},
        {"criterion": "Supporting: same coverage check, full variant",
         "threshold": "interval contains 0.90",
         "measured": f"{cvf['hits']}/{cvf['n']} = {cvf['coverage']:.2f}; CP "
                     f"[{cvf['clopper_pearson_95_neff'][0]:.3f}, "
                     f"{cvf['clopper_pearson_95_neff'][1]:.3f}]",
         "passed": bool(cvf["passed_prereg_rule"])},
        {"criterion": "Denton keeps zero months zero, no negative month, annual totals exact "
                      "(both variants, Denton series and all 200 draws per district)",
         "threshold": "0 negative months; max abs(annual sum - total) < 1e-9",
         "measured": "negative months 0; max abs diff "
                     + f"{max(ex[v][d][k] for v in ex for d in ('changhua', 'yunlin') for k in ('denton_annual_max_abs_diff', 'draws_annual_max_abs_diff')):.1e}",
         "passed": all(ex[v][d]["negative_months_denton"] == 0 and
                       ex[v][d]["negative_months_draws"] == 0 and
                       ex[v][d]["denton_annual_max_abs_diff"] < 1e-9 and
                       ex[v][d]["draws_annual_max_abs_diff"] < 1e-9
                       for v in ex for d in ("changhua", "yunlin"))},
        {"criterion": "Pre-origin variant fitted on 2012-2022 only (U3): outputs end by "
                      "2022-12; beta, a, QC medians from 2012-2022",
         "threshold": "max output date <= 2022-12-31",
         "measured": ex["preorigin"]["max_date_in_outputs"],
         "passed": ex["preorigin"]["max_date_in_outputs"] <= "2022-12-31"},
        {"criterion": "Monthly values from the daily table; printed monthly mean differs by "
                      "> 5 % (reported)",
         "threshold": "reported (plan: 23 station-months in the scout's parse)",
         "measured": f"{pq['printed_monthly_off_gt_tol']} of {pq['printed_monthly_compared']} "
                     "station-months",
         "passed": True},
        {"criterion": "Parse re-validated against the scout's daily table",
         "threshold": "0 differing station-days",
         "measured": f"{pvp['different']} of {pvp['compared']} differ; {pvp['only_in_w1']} "
                     "station-days only in W1 (H064 2012, day 1 of each month, missed by the "
                     "scout)",
         "passed": pvp["different"] == 0},
        {"criterion": "Aggregates re-read from L0: Jiji CSV vs HTML page; Jan-May table by "
                      "pdftotext vs the scout's hand transcription",
         "threshold": "0 differences",
         "measured": f"Jiji {len(com['jiji_check']['cell_differences'])} cell differences "
                     f"over {com['jiji_check']['html_rows']} rows; Jan-May max abs diff "
                     f"{com['janmay_vs_pilot']['max_abs_diff_1e8m3']:.1e}",
         "passed": com["jiji_check"]["passed"] and
                   com["janmay_vs_pilot"]["max_abs_diff_1e8m3"] < 1e-9},
        {"criterion": "Spatial allocation reused from v3 reproduces v3's own field",
         "threshold": "max relative error < 1e-9",
         "measured": f"{P['allocation']['v3_rebuild_max_rel']:.1e}",
         "passed": P["allocation"]["v3_rebuild_max_rel"] < 1e-9},
        {"criterion": "Manifests re-hash (A0 checker) and loader guard enforces the uses",
         "threshold": "0 problems; forbidden uses refused",
         "measured": "; ".join(f"{v}: {c['status']} ({c['n_hashed']} files)"
                               for v, c in ver["checker"].items())
                     + "; preorigin refuses scoring_target, calibration_target, "
                       "forcing_recorded_arm and the LUIMAP grid for 'selection'; full refuses "
                       "forcing_frozen, model_design, scoring_target",
         "passed": all(c["status"] == "ok" for c in ver["checker"].values())},
    ]
    status = "partial"
    summary = (
        "W1 rebuilt the monthly Jiji-weir canal deliveries (surface water v4) from L0: 14 "
        "yearbook PDFs parsed with pdftotext (123 station-years, monthly values from the "
        "daily table), a daily inflow composite (H075+H049+H064) with per-year gauge QC and "
        "donor swaps rescaled to the weir's published inflow, the supply-limited diversion "
        "indicator, proportional first-difference Denton benchmarking to the annual district "
        "totals (0 negative months, totals exact), 200 state-space posterior draws per "
        "district (log indicator + AR(1), exact annual totals, Jan-May and MOA crop-share "
        "aggregates), and the v3 farmland allocation onto the twin grid (L3 field plus a "
        "v3-level timing-only variant). Products: surface_water_v4/preorigin/v1 (beta, "
        "demand scales and QC medians fitted on 2012-2022, Denton to 2022; LUIMAP 2024 "
        "listed as the U3 exemption for the grid field only) and surface_water_v4/full/v1 "
        "(2012-2025, recorded-forcing arm only). On the pre-registered Jan-May "
        f"leave-one-year-out benchmark the pre-origin variant scores RMSE "
        f"{f3(bp['scores']['supply_limited_denton']['chc']['rmse'])} / "
        f"{f3(bp['scores']['supply_limited_denton']['yl']['rmse'])} x 10^8 m3 (Changhua / "
        f"Yunlin) against climatology {f3(bp['scores']['clim_fraction_loo']['chc']['rmse'])} / "
        f"{f3(bp['scores']['clim_fraction_loo']['yl']['rmse'])} and v3 "
        f"{f3(bp['scores']['v3']['chc']['rmse'])} / {f3(bp['scores']['v3']['yl']['rmse'])}; all "
        "four 90 % paired-bootstrap intervals exclude zero, so the primary acceptance passes, "
        "but narrowly (Yunlin vs climatology upper bound "
        f"{bp['compare']['clim_fraction_loo']['yl']['boot_hi']:+.3f}) and the sign tests are "
        "not significant (best p = 0.125). The full variant fails the same rule "
        f"(RMSE {f3(bf['scores']['supply_limited_denton']['chc']['rmse'])} / "
        f"{f3(bf['scores']['supply_limited_denton']['yl']['rmse'])}; 2 of 4 intervals include "
        "zero) and ships labelled 'not distinguishable from climatology on the benchmark'. "
        "The benchmark is design-time-contaminated: the scout scored the same seven years "
        "before the design was fixed, and no fresh Jan-May units exist. The pre-registered "
        "supporting head check fails in one of four groups (Yunlin layer 1, "
        f"{hc['yunlin_L1']['v4_minus_v3_partial']:+.4f}); the state-space intervals pass the "
        f"weak coverage rule but cover only {cvp['hits']}/{cvp['n']} held-out values. Plan "
        "step 6 (2026 rainfall-runoff extension) and the forward archive of the real-time "
        "feeds were not done, hence status partial. No GPU, 0 Claude API tokens, nothing "
        "under hydrophysics/, tests/ or L0 written, nothing staged in git.")
    deviations = [
        "Code in dataprep_staging/w1_canal/ (user decision D12), not results/dataprep/_staging_code/.",
        "Pre-registration W1_janmay_benchmark_2026-10-12 was written at 02:40:16Z by W1's own "
        "writer (w1_canal/prereg.py), two minutes before the A1 tool (harness/prereg.py, "
        "02:42Z) existed, so it is not in the A1 INDEX and the A1 tool cannot gate it. The "
        "looks were logged in the A1 look log afterwards: a back-dated design-time seed for "
        "the scout (2026-10-11T16:56Z), the first W1 session's dry run (02:45-02:46Z, after "
        "the pre-registration, logged late) and the product build (06:16Z).",
        "The registered test was first run as a dry run by the first W1 session; that "
        "session was interrupted before publishing. This session rebuilt from L0 with the "
        "registered code unchanged (all 8 registered code files and 19 inputs still match "
        "the pre-registration hashes); every benchmark, coverage and monthly output is "
        "byte-identical to the dry run. Only checks.py and build.py gained two reporting "
        "checks (comparison with the scout's parse and transcription) and repo-relative "
        "paths; looks.py, verify.py and report.py are new.",
        "Plan step 5 says annual totals 2006-2025: the indicator needs the yearbook gauges, "
        "which start in 2012, so the state space uses 2012-2022 (pre-origin) and 2012-2025 "
        "(full). Crop-season splits: MOA 691 crop-1 shares 2012-2015 only; WRA 58701 "
        "(per crop to 2020) was not used (unit glitches in the association 12/13 rows).",
        "Plan step 6 (2026 extension by a rainfall-runoff regression from CWA CODIS mountain "
        "stations) not done: no mountain-station rain is in L0 (the CODIS archive holds the "
        "26 fan gauges only), it needs a new L0 fetch, and U4 lists canal v4 for 2023-25 only "
        "in the recorded-forcing arm, so no Phase-0/1 consumer is blocked. The forward "
        "archive of the WRA real-time discharge and weir feeds needs a recurring job (a user "
        "decision) and was not started.",
        "Plan step 7 (basis for P4 E4) is superseded by P4m; the crop-calendar delivery "
        "table percolation_basis.csv is still provided.",
        "Gate semantics: manifest gate.passed means 'ships' (point RMSE below both baselines "
        "in both districts); acceptance_passed and label carry the pre-registered decision, "
        "so the full variant ships with gate.passed = true, acceptance_passed = false and "
        "its label, as the plan prescribes.",
        "The structural-zero months (Dec-Jan, winter weight 0 chosen by leave-one-year-out) "
        "carry as sd the Denton value of the winter-0.5 alternative, not a posterior sd.",
        "The pre-origin product's validation.json repeats the parse QC over all yearbook "
        "years 2012-2025 (reporting only; nothing fitted for the pre-origin variant used "
        "2023+ data).",
    ]
    issues = [
        "Level: v4 is at the Jiji weir level (district supply). v3 was at the MOA 2012-2019 "
        f"level, {P['allocation']['k_v3level']['chc']:.3f} x (Changhua) and "
        f"{P['allocation']['k_v3level']['yl']:.3f} x (Yunlin) higher, so swapping v4 into the "
        "twin changes the delivered volume and the Changhua/Yunlin balance. grid_sw_v4.npz "
        "carries value_v3level (v4 timing at v3's annual totals) for a timing-only A/B in "
        "I1 rung R3; the calibrated log_sw_scale was fitted on v3's level.",
        "The benchmark is weak evidence: 7 years per district, the scout already scored them, "
        "the percentile-bootstrap interval at n = 7 tends to be too narrow, sign tests are not "
        "significant (5/7, 4/7 and 6/7 wins), and the Yunlin-vs-climatology bound is "
        f"{bp['compare']['clim_fraction_loo']['yl']['boot_hi']:+.3f}. v4 over-delivers Jan-May "
        f"on average (bias {bp['scores']['supply_limited_denton']['chc']['bias']:+.3f} / "
        f"{bp['scores']['supply_limited_denton']['yl']['bias']:+.3f}). A definitive test needs "
        "the monthly intakes (request R-c, not drafted per user decision).",
        "The un-benchmarked raw indicator scores slightly better on Jan-May than the Denton "
        f"series ({f3(bp['scores']['supply_limited_raw_indicator']['chc']['rmse'])} / "
        f"{f3(bp['scores']['supply_limited_raw_indicator']['yl']['rmse'])} vs "
        f"{f3(bp['scores']['supply_limited_denton']['chc']['rmse'])} / "
        f"{f3(bp['scores']['supply_limited_denton']['yl']['rmse'])}); not used to choose "
        "anything (post hoc).",
        "Full variant: the post-2022 gauges disagree with the weir inflow (annual scale "
        f"factors 2024 {F['composite']['scale_to_weir_inflow']['2024']:.3f}, 2025 "
        f"{F['composite']['scale_to_weir_inflow']['2025']:.3f}; H049 2024 donor-swapped), and "
        "QC medians fitted on 2012-2025 swap H064 in 2020 and 2022 as well; beta rises from "
        f"{P['final_fit']['beta']:.3f} to {F['final_fit']['beta']:.3f} and the annual "
        f"indicator R2 falls ({P['final_fit']['indicator_annual_r2']['chc']:.2f}/"
        f"{P['final_fit']['indicator_annual_r2']['yl']:.2f} -> "
        f"{F['final_fit']['indicator_annual_r2']['chc']:.2f}/"
        f"{F['final_fit']['indicator_annual_r2']['yl']:.2f}). Use the full variant only in the "
        "recorded-forcing arm (U4), with its label.",
        "State-space draws (district_draws.npz, 200 per district, equal weight, districts "
        "drawn independently) are the forcing members for 4.8. Their 90 % intervals covered "
        f"{cvp['hits']}/{cvp['n']} held-out Jan-May totals (all misses were over-predictions: "
        "2015 both districts, 2020 Yunlin); the coverage rule passes only because n_eff = 7 "
        "makes it weak. Treat the intervals as at best roughly calibrated, likely too narrow "
        "in drought years.",
        "LUIMAP 2024 exemption (pre-origin grid field only): spreading the same in-fan volume "
        f"by county area instead of farmland changes mean cell depth with correlation "
        f"{sens['corr_mean_depth']:.3f}, median |relative change| "
        f"{sens['median_abs_rel_diff']:.3f}; the district series and draws do not depend on "
        "LUIMAP and are fully pre-origin.",
        "The head supporting check failed in Yunlin layer 1 by "
        f"{hc['yunlin_L1']['v4_minus_v3_partial']:+.4f} (other groups +0.024 to +0.042); the "
        "inflow-only series is about as good there (partial r "
        f"{hc['yunlin_L1']['inflow_only']['partial_r_given_rain3']:.3f} vs v4 "
        f"{hc['yunlin_L1']['v4_preorigin_denton']['partial_r_given_rain3']:.3f}). The physical "
        "A/B test is I1 rung R3.",
        "Crop-1 share vs MOA 691 (not gated): Changhua v4 0.41-0.51 vs MOA 0.37-0.48; Yunlin "
        "v4 0.22-0.30 vs MOA 0.36-0.44 (MOA counts the whole office, all sources); the "
        "chosen Yunlin crop-1 demand share is 0.30.",
        "The manifests list dataprep_staging/harness/coverage.py (A1, still being edited) "
        "as code, used only to report the A1 coverage gate; if A1 changes it the checker "
        "will flag these manifests (provenance drift, not corruption). Editing any listed "
        "w1_canal file also requires a v2.",
        "Licence: everything derived from the yearbooks stays under results/ (copying "
        "disallowed, plan 2.6; decision D3).",
        "2026 deliveries and the real-time archive: see deviations; a recipe is a CODIS "
        "fetch of the upper-basin stations (L0 with SOURCE.json), a monthly regression of "
        "the composite on basin rain fitted on 2012-2022, and indicator values flagged 6 "
        "with the annual leave-one-year-out error.",
    ]
    key_numbers = [
        f"Pre-origin Jan-May LOO RMSE (10^8 m3): v4 {f3(bp['scores']['supply_limited_denton']['chc']['rmse'])}/"
        f"{f3(bp['scores']['supply_limited_denton']['yl']['rmse'])}, climatology "
        f"{f3(bp['scores']['clim_fraction_loo']['chc']['rmse'])}/{f3(bp['scores']['clim_fraction_loo']['yl']['rmse'])}, "
        f"v3 {f3(bp['scores']['v3']['chc']['rmse'])}/{f3(bp['scores']['v3']['yl']['rmse'])}, calendar "
        f"{f3(bp['scores']['calendar_denton']['chc']['rmse'])}/{f3(bp['scores']['calendar_denton']['yl']['rmse'])} "
        "(Changhua/Yunlin)",
        "Pre-origin RMSE differences [90 % paired bootstrap]: vs clim chc "
        f"{boot(bp, 'clim_fraction_loo', 'chc')}, yl {boot(bp, 'clim_fraction_loo', 'yl')}; vs v3 "
        f"chc {boot(bp, 'v3', 'chc')}, yl {boot(bp, 'v3', 'yl')}",
        f"Full Jan-May LOO RMSE: v4 {f3(bf['scores']['supply_limited_denton']['chc']['rmse'])}/"
        f"{f3(bf['scores']['supply_limited_denton']['yl']['rmse'])}; vs clim yl "
        f"{boot(bf, 'clim_fraction_loo', 'yl')}, vs v3 chc {boot(bf, 'v3', 'chc')} -> label "
        "'not distinguishable from climatology on the benchmark'",
        f"Fit pre-origin: beta {P['final_fit']['beta']:.3f}, s_chc {P['final_fit']['s_chc']}, "
        f"s_yl {P['final_fit']['s_yl']}, winter {P['final_fit']['winter']}, annual fit rms "
        f"{P['final_fit']['annual_fit_rms_1e8m3']:.3f}; full: beta {F['final_fit']['beta']:.3f}",
        f"State-space LOO 90 % coverage: pre-origin {cvp['hits']}/{cvp['n']}, full "
        f"{cvf['hits']}/{cvf['n']}; mean 90 % width {cvp['mean_width_90']['chc']:.2f}/"
        f"{cvp['mean_width_90']['yl']:.2f} x 10^8 m3",
        "Head partial r given rain, v4 - v3: " + ", ".join(
            f"{g} {hc[g]['v4_minus_v3_partial']:+.4f}" for g in hc),
        f"Parse: {pq['station_years']} station-years, {pq['printed_monthly_off_gt_tol']}/"
        f"{pq['printed_monthly_compared']} printed monthly means off > 5 %, 0/"
        f"{pvp['compared']} station-days differ from the scout, +{pvp['only_in_w1']} recovered",
        f"In-fan fraction (v3 farmland): Changhua {P['allocation']['infan_fraction']['chc']:.3f}, "
        f"Yunlin {P['allocation']['infan_fraction']['yl']:.3f}; v3/v4 level ratio "
        f"{P['allocation']['k_v3level']['chc']:.3f} / {P['allocation']['k_v3level']['yl']:.3f}",
        "Claude API tokens: 0; GPU: none",
    ]
    products = [
        {"path": rel_to_repo(pre), "layer": "L2 (district series, draws, daily composite) + L3 "
                                          "grid array",
         "manifest": rel_to_repo(pre / "MANIFEST.json"),
         "allowed_uses": "forcing_frozen, model_design, selection, prior, prior_width_input, "
                         "design_time_evidence, after_the_fact_analysis (grid_sw_v4.npz only "
                         "via the LUIMAP 2024 exemption: forcing_frozen, model_design, "
                         "design_time_evidence)",
         "fit_window_end": "2022-12 (non-exempt; 2024-12 with the LUIMAP exemption for the "
                           "grid field)"},
        {"path": rel_to_repo(full), "layer": "L2 + L3 grid array",
         "manifest": rel_to_repo(full / "MANIFEST.json"),
         "allowed_uses": "forcing_recorded_arm, statistical_layer_history, "
                         "after_the_fact_analysis (labelled 'not distinguishable from "
                         "climatology on the benchmark')",
         "fit_window_end": "2025-12"},
    ]
    out = {"wp": "W1", "status": status, "date": "2026-10-12", "summary": summary,
           "acceptance": acceptance, "products": products,
           "code_paths": [rel_to_repo(C.HERE / f) for f in CODE],
           "key_numbers": key_numbers, "deviations_from_plan": deviations,
           "issues_for_downstream": issues,
           "preregistration": f"results/dataprep/preregistration/{PREREG_ID}.json",
           "evidence": ["results/dataprep/reports/W1_evidence/verify.json",
                        rel_to_repo(pre / "validation.json"), rel_to_repo(full / "validation.json"),
                        rel_to_repo(pre / "QC_REPORT.md"), rel_to_repo(full / "QC_REPORT.md")],
           "ledger": {"fragment": "results/dataprep/ledger/rows/W1.csv",
                      "unledgered_cited_numbers": missing, "conflicts_all_fragments":
                      len(conflicts)},
           "claude_api_tokens": 0, "gpu": "none"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "W1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    md = ["# WP W1: monthly canal deliveries, surface water v4 (report, 2026-10-12)", "",
          f"Status: **{status}**. {summary}", "",
          "## Products", ""]
    for p in products:
        md.append(f"- `{p['path']}/` ({p['layer']}); allowed: {p['allowed_uses']}; "
                  f"fit_window_end {p['fit_window_end']}.")
    md += ["", "Each holds `district_monthly.csv` (value = Denton series, sd = state-space "
           "posterior sd, flag 5; state-space mean and 5/50/95 % columns; the indicator), "
           "`district_draws.npz` (200 draws per district), `inflow_composite_daily.parquet` "
           "(value, sd, flag 1/3/4, per-gauge values and flags), `gauges_daily_L1.parquet` "
           "(all 9 Choushui gauges, flags 0/8/9), `grid_sw_v4.npz` (value, sd, flag, dates, "
           "mask hash, grid hash, per-cell shares, `value_v3level`), `percolation_basis.csv`, "
           "`janmay_validation.csv`, `statespace_loo_coverage.csv`, the printed monthly and "
           "annual means, `validation.json` and `QC_REPORT.md`.", "",
           "## Acceptance", "", "| criterion | threshold | measured | pass |",
           "|---|---|---|---|"]
    for a in acceptance:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'yes' if a['passed'] else 'NO'} |")
    md += ["", "## Jan-May benchmark, pre-origin variant (primary, leave-one-year-out "
           "2015-2021, 10^8 m3)", ""] + bench_table(bp) + [""] + compare_table(bp)
    md += ["", "## Jan-May benchmark, full variant (secondary)", ""] + bench_table(bf) + [""] \
        + compare_table(bf)
    md += ["", "Interpretation (post hoc, written after the results were known, not tested): the "
           "pre-origin variant meets the registered rule, the full variant does not. The gap is "
           "consistent with the post-2022 gauges, which disagree with the weir's inflow and "
           "shift beta and the QC medians (H064 swapped in 2020 and 2022 only in the full "
           "variant). Both results rest on the same 7 design-time years.",
           "", "## Deviations from the plan", ""] + [f"- {x}" for x in deviations]
    md += ["", "## Issues for downstream WPs", ""] + [f"- {x}" for x in issues]
    md += ["", "## Key numbers", ""] + [f"- {x}" for x in key_numbers]
    md += ["", "## Provenance", "",
           f"- Pre-registration: `results/dataprep/preregistration/{PREREG_ID}.json` "
           "(02:40:16Z); looks logged in `results/dataprep/preregistration/looks.jsonl`.",
           f"- Ledger: `results/dataprep/ledger/rows/W1.csv`; cited numbers without a row: "
           f"{missing}; conflicts across all fragments: {len(conflicts)}.",
           "- Evidence: `results/dataprep/reports/W1_evidence/verify.json` (checker, guard "
           "probes, exactness); per-variant `validation.json` and `QC_REPORT.md`.",
           "- Code: " + ", ".join(f"`{rel_to_repo(C.HERE / f)}`" for f in CODE) + ".",
           "- Claude API tokens: 0. GPU: none. Nothing under hydrophysics/, tests/ or L0 "
           "written; nothing staged in git.", ""]
    (REPORTS / "W1.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    o = main()
    print(o["status"], o["ledger"])
    for a in o["acceptance"]:
        print("yes" if a["passed"] else "NO ", "|", a["criterion"][:70], "|", a["measured"][:110])

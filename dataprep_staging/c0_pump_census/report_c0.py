"""WP C0: write results/dataprep/reports/C0.json (acceptance table, products, deviations, issues)
from the evidence files. The prose report C0.md is written alongside by hand from the same files.

Usage: python -I report_c0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

CK = lib.EVIDENCE / "checks"
PE = "results/dataprep/pump_electricity"

CODE_FILES = ("lib.py", "build.py", "assemble.py", "checks.py", "finalize.py", "looks_c0.py",
              "prereg_c0.py", "diagnostics.py", "ledger_c0.py", "resume_c0.py",
              "report_ledger_c0.py", "report_c0.py")

SUMMARY = (
    "C0 builds the cleaned Taiwan Power pump-electricity census for every county with pumps in "
    "the twin's active cells (Yunlin, Changhua and the 320 in-grid Chiayi pumps), from the new "
    "L0 parquet derivative only, cleaned with hydrophysics.twin.pumping.clean_census exactly as "
    "the model cleans Yunlin (shared meters once, HP-share split, duty cap 1.0 on 2012-2022). "
    "It delivers an L1 census table without personal columns (census_L1/v1) and two L2 "
    "products (preorigin/v1 to 2022-12, full/v1 to 2026-07 with 2026-07 masked). Each holds "
    "per-pump and per-cell x class monthly kWh with flags, per-meter billing phases, "
    "pre-install structural zeros, terminated-meter labels and the domestic-excluded "
    "sensitivity variant. All 9 gating tests of the pre-registration C0.census_product.v1 "
    "pass, and all three product gates pass. The model's own path and the C0 pipeline "
    "reproduce the calibrated model's cached Yunlin forcing to 1.5e-10 kWh (T1a/T1b/T2a). The "
    "delivered Yunlin part differs from it only by one irrigation pump's readings missing from "
    "AMP_V2: 6,782 kWh over 2012-2022, at most 1.7e-5 of any month's total (T2b). "
    "In-grid cleaned electricity 2012-2022 is 188.5 GWh/yr (Yunlin), 187.4 (Changhua) and 0.63 "
    "(Chiayi). Changhua's domestic class is 23.0 % of its in-grid energy against 5.6 % in "
    "Yunlin. The census covers 0.39-0.45 (Changhua) and 0.43-0.55 (Yunlin) of TPC "
    "farm-sector sales in 2016-2025, and the coverage is not stationary. This comparison is "
    "descriptive, because P1's TPC product fails its own gate. Two choices are recorded as "
    "deviations: the pre-install coding is kept as delivered rather than harmonised (+0.35 % "
    "of Changhua kept energy under the alternative), and a phase-B meter's December 2022 "
    "value holds half of a bill issued in 2023 (the model's convention). The work was resumed "
    "after a usage-limit interruption; a read-only resume check re-hashed all 77 recorded "
    "files and found the pre-registration intact. No GPU, no Claude API, nothing written under "
    "hydrophysics/, tests/ or L0, nothing staged in git.")

PRODUCTS = [
    {"path": f"{PE}/census_L1/v1", "layer": "L1",
     "manifest": f"{PE}/census_L1/v1/MANIFEST.json (+ PRODUCT_CARD.md, validation.json)",
     "allowed_uses": "model_design, prior, prior_width_input, design_time_evidence, validator, "
                     "after_the_fact_analysis (attributes + 2012-2022 duty/kept; no kWh)",
     "fit_window_end": "2022-12"},
    {"path": f"{PE}/preorigin/v1", "layer": "L2",
     "manifest": f"{PE}/preorigin/v1/MANIFEST.json (+ PRODUCT_CARD.md, validation.json)",
     "allowed_uses": "forcing_frozen (calibration and frozen forecasts), model_design, prior, "
                     "prior_width_input, selection, design_time_evidence, validator, "
                     "scoring_target, after_the_fact_analysis",
     "fit_window_end": "2022-12"},
    {"path": f"{PE}/full/v1", "layer": "L2",
     "manifest": f"{PE}/full/v1/MANIFEST.json (+ PRODUCT_CARD.md, validation.json)",
     "allowed_uses": "forcing_recorded_arm (U4 recorded-forcing diagnostic arm), "
                     "statistical_layer_history, validator, after_the_fact_analysis",
     "fit_window_end": "2026-07"},
]

DELIVERABLES = [
    ("(a) L1 census table, no personal columns, county column, harmonised names",
     "census_L1/v1/census_pumps.parquet: 243,033 rows (Changhua 125,941, Yunlin 116,769, "
     "Chiayi 323 = 320 in grid + 3 sharing their meters), 31 columns; T8 0 violations"),
    ("(b) L2 per-cell x class monthly kWh (value, flag, sd), billing phase per meter, masks",
     "cells_by_class.npz value/flag/sd (6 x 2148 x T) + value_by_county; meters_<county>.parquet "
     "phase_code/phase_detected; pre_install structural zeros; 2026-07 phase-A NaN flag 8; "
     "terminated meters labelled; T6 0 violations"),
    ("(c) domestic-excluded sensitivity variant",
     "cells_total.npz value_nodomestic/flag_nodomestic; county_class_monthly 'all_nodomestic'; "
     "identity all - domestic = nodomestic within 6e-11 kWh (resume spot check)"),
    ("(d) Yunlin part reproduces the model's cleaned Yunlin forcing 2012-2022",
     "T1a/T1b/T2a/T2b passed (see rows above)"),
    ("(e) validation against the TPC county farm-sector aggregate",
     "T4 (descriptive): annual and month-of-year coverage, Changhua/Yunlin ratios, monthly "
     "correlations at lags -1/0/+1; reproduces P1 C8 to 7.8e-16"),
    ("(f) product card", "PRODUCT_CARD.md in each of the three version directories"),
]

DEVIATIONS = [
    "Pre-install coding is not harmonised across counties (P1 asked C0 to choose one coding "
    "for both). The duty statistic sees the L0 readings as delivered: Yunlin's TPC-coded "
    "pre-install zeros enter its mean (model-identical), and Changhua/Chiayi NaN do not. "
    "Coding every pre-install month 0 would flip 2 Yunlin meters (+0.03 % of 2012-2022 kept "
    "kWh) and break the exact reproduction of the calibrated forcing (task d). For Changhua it "
    "would keep 64 more meters (+0.35 %); Chiayi does not change. Coding them all NaN would "
    "also change Yunlin. In the delivered arrays every pre-install month is coded the same way "
    "in every county: value 0, flag 0, method pre_install, never imputed. The choice was made "
    "knowing the sensitivity and is disclosed in the pre-registration.",
    "Task (e) asked for validation with the P1 products as validators. P1's tpc_observed/v1 "
    "fails its gate (G3) and census_check/v2 fails its construction gate (C9), so the guard "
    "refuses their validator use. T4 (TPC coverage) and T3 (agreement with P1's census truth) "
    "therefore open them as after_the_fact_analysis: T4 is descriptive and T3 is reported. "
    "Neither is part of any gate.",
    "The pre-origin variant keeps the model's billing convention. A phase-B (M1) meter's "
    "December value is half of a December-January bill read in 2023, so the 2022-12 forcing "
    "of those meters (Yunlin 43,834, Changhua 50,820 meters) carries half a month of January "
    "2023 electricity. It is disclosed, not masked, because masking would break the identity "
    "with the calibrated forcing.",
    "Bimonthly bills keep TPC's even split over their two months (the model's convention). "
    "The P2a Chow-Lin split is not applied, because P2 builds on C0. In the full variant, from "
    "2025-07 on, 12-33 % of pairs post the whole bill to the second month; an even-split "
    "alternative (flag 1) is shipped next to the primary arrays.",
    "'sd where applicable': every value is an observed reading or a structural zero, so sd is "
    "NaN throughout. Missing post-install readings are flagged 8, never filled, and count as "
    "zero in aggregates, as in the model. A cell aggregate is flagged 9 when more than 10 % of "
    "its installed kept HP is missing.",
    "Order of work: the products were built before the pre-registration and are pinned by hash "
    "in it. Ten design-time looks were logged and disclosed, among them the six 2012-2022 "
    "class totals of the cached model forcing. No comparison with any reference was computed "
    "before registration. The tests are deterministic construction and reproduction checks, "
    "not out-of-sample tests.",
    "Resumed WP: the earlier session stopped after scoring, diagnostics and ledger rows but "
    "before the report. This session rebuilt and re-scored nothing. It ran a read-only resume "
    "check (resume_c0.py, logged as [C0-resume-1]), added report ledger rows "
    "(report_ledger_c0.py) and wrote the report. The build cache "
    "results/dataprep/pump_electricity/_build_c0/ has no manifest and is not a product.",
]

ISSUES = [
    "Model integration (after the code freeze): the calibrated model's forcing (E_by_class "
    "from AMP_V2) holds Yunlin pumps only. Adding Changhua from preorigin/v1 roughly doubles "
    "the metered in-grid pumping energy (+187.4 GWh/yr on 188.5). Consuming the product needs "
    "a change under hydrophysics/twin/ (frozen while batch 8 runs) and a recalibration on the "
    "pre-origin variant (U1). The K_pump, theta and column fitted on Yunlin-only forcing do not "
    "transfer. C0 does not depend on theta, so batch 8 being provisional does not affect it.",
    "Changhua domestic class: 23.0 % of Changhua in-grid energy 2012-2022 (43.1 GWh/yr) against "
    "5.6 % in Yunlin (10.5). The duty cap already drops 6,100 of 21,506 Changhua domestic "
    "meters and keeps 538 of 5,502 de-duplicated GWh (2012-2022), a sign of heavy non-pump "
    "load. Both counties' domestic classes peak in August (Aug/Jan 1.41 Changhua, 1.50 "
    "Yunlin), so the seasonal shape alone does not separate household load from pumping. Use "
    "cells_total value_nodomestic as the sensitivity member. Like for like (both counties "
    "without domestic) the Changhua/Yunlin census ratio is 0.76-0.81 in 2017-2022.",
    "Pre-install coding sensitivity (deviation 1): if P4m wants a member with every "
    "pre-install month coded 0, it adds 64 Changhua meters / 65 pumps (+0.35 % of Changhua "
    "2012-2022 kept kWh). It needs a new product version (v2); v1 is immutable.",
    "Guard filtering: guard.read_table for forcing_frozen keeps flag-0 rows only, so it drops "
    "the flag-9 rows of county_class_monthly (aggregates with more than 10 % of installed kept "
    "HP missing). Consumers of the npz arrays must apply the flags themselves. Flag-9 class x "
    "cell-months hold 0.38 % (Yunlin), 0.30 % (Changhua) and 0.42 % (Chiayi) of 2012-2022 "
    "in-grid energy.",
    "Terminated-meter labels in the pre-origin variant are provisional for runs that start in "
    "2021-2022, because a later restart cannot be known before the origin. For example, "
    "Yunlin pre-origin labels 1,520 meters terminated from 2021, against 722 in the full "
    "variant. The label is a method code and per-meter column only; values stay observed "
    "zeros, so the forcing does not change.",
    "Recorded-forcing arm (full/v1): from 2025-07, 12-33 % of bimonthly pairs post the whole "
    "bill to the second month. The even-split alternative changes monthly fan totals by a mean "
    "1.9 % in 2025-07..2026-06 (0.013 % before) and re-splits 77 % of the cell-months of that "
    "year. Consider it for consistency with earlier years. 2026-07 is masked: 3,635 class x "
    "cell entries are flag 8 (phase-A pairs incomplete) and 1,571 flag 9.",
    "TPC coverage is not stationary and differs between counties. Census/TPC 農林漁牧 (county, "
    "all classes): Changhua 0.386-0.450, Yunlin 0.434-0.547 in 2016-2025. The Changhua/Yunlin "
    "census ratio is 0.93-1.03 (2017-2025) against 1.13-1.20 for TPC farm sales, so TPC's "
    "farm sector carries relatively more non-census load in Changhua. Do not scale Changhua "
    "pumping by TPC. Month-of-year coverage peaks in March-April (Changhua 0.51, Yunlin 0.65). "
    "Monthly r of census against TPC at lag 0 for 2016-2022: Changhua 0.79 raw / 0.75 "
    "deseasonalised, Yunlin 0.52 raw / 0.81 deseasonalised; Yunlin's raw r is higher at lag +1 "
    "(0.62). The TPC billing month is undocumented.",
    "The census is frozen at mid-2016 installs, so no post-2016 pump appears; the same holds "
    "for Yunlin in the model. P1 found non-zero meters falling a median 14 % (Yunlin) and 10 % "
    "(Changhua) year on year in 2026. Check billing-group completeness before reading a "
    "2025-26 decline as less pumping.",
    "Privacy: census_L1 carries pump coordinates (TWD97) and well numbers; the per-pump L2 "
    "arrays carry well numbers. Both are gitignored, and the manifests declare personal_data. "
    "No name, address, postal code or raw meter number is written anywhere (T8). Never copy "
    "these files into committed paths.",
    "Provenance: the C0 manifests hash the L0 parquet files directly, so they verify even "
    "though the TPC SOURCE.json uses files[].path, which common/source.py does not read "
    "(A0fix issue, owner edit). ledger/data_roles.csv was not reassembled (shared file). "
    "rows/C0.csv is complete, with 0 conflicts across all fragments.",
    "Chiayi: 316 kept in-grid pumps, 0.63 GWh/yr (0.17 % of the in-grid total). They are "
    "included in the cell arrays; county-scope rows of county_class_monthly cover all 26,056 "
    "kept Chiayi pumps. Pingtung has 0 pumps in active cells and is excluded.",
]


def main() -> int:
    t = {k: lib.load_json(CK / f"{k}.json") for k in ("T1a", "T1b", "T2a", "T2b", "T3", "T4",
                                                     "T5", "T6", "T7", "T8", "T9")}
    s = lib.load_json(CK / "summary.json")
    dg = lib.load_json(lib.EVIDENCE / "diagnostics.json")
    t1a_max = max(v["max_abs_delta"] for v in t["T1a"]["per_class"].values())
    c8 = t["T4"]["p1_c8_reproduction"]
    acc = [
        {"criterion": "T1a model path re-run on AMP_V2 reproduces the cached calibrated forcing "
                      "(E_by_class, inputs.local.pkl), same grid/dates/classes",
         "threshold": "every element |d| <= 1e-6 kWh + 1e-9 |ref|",
         "measured": f"max |d| {t1a_max} kWh over 6 classes x 2148 cells x 132 months; grid, "
                     "dates, class keys equal", "passed": t["T1a"]["pass"]},
        {"criterion": "T1b C0 pipeline on AMP_V2 reproduces the cached forcing",
         "threshold": "every element |d| <= 1e-6 kWh + 1e-9 |ref|",
         "measured": f"max |d| {t['T1b']['max_abs_delta']:.3g} kWh", "passed": t["T1b"]["pass"]},
        {"criterion": "T2a C0 pipeline on the new L0 Yunlin sheet (AMP-absent readings masked) "
                      "reproduces the cached forcing",
         "threshold": "every element |d| <= 1e-6 kWh + 1e-9 |ref|",
         "measured": f"max |d| {t['T2a']['max_abs_delta']:.3g} kWh; "
                     f"{t['T2a']['pump_months_absent_in_amp']} AMP-absent pump-months "
                     f"({t['T2a']['absent_kwh_2012_2022']:.0f} kWh in 2012-2022)",
         "passed": t["T2a"]["pass"]},
        {"criterion": "T2b delivered Yunlin part (pre-origin, new L0) reproduces the model's "
                      "cleaned Yunlin forcing 2012-2022 'to within rounding' (task d)",
         "threshold": "(a) |monthly fan-total delta| <= 1e-4 of the month's total; (b) sum |d| "
                      f"<= {t['T2b']['b_limit_kwh']:.0f} kWh (2 x AMP-absent kWh + 1)",
         "measured": f"(a) max {t['T2b']['a_max_rel_month_total_delta']:.3g} "
                     f"({t['T2b']['a_worst_month']}); (b) {t['T2b']['b_sum_abs_delta_kwh']:.1f} "
                     f"kWh; max element {t['T2b']['max_element_abs_delta_kwh']} kWh",
         "passed": t["T2b"]["pass"]},
        {"criterion": "T3 (reported, not gating) county / in-grid class x month sums equal P1 "
                      "census_check/v2 truth (both counties, both variants)",
         "threshold": "|d| <= 1e-6 GWh",
         "measured": f"max |d| {t['T3']['variants']['full']['max_abs_delta_GWh']:.3g} GWh; 0 "
                     "unmatched rows (4,608 pre-origin, 5,616 full)", "passed": t["T3"]["pass"]},
        {"criterion": "T4 (descriptive) TPC coverage; reproduction of P1 C8 cov_clean_all",
         "threshold": "descriptive; C8 reproduction <= 1e-6 relative",
         "measured": f"C8 max rel diff {c8['max_rel_diff']:.2g}; coverage Changhua 0.386-0.450, "
                     "Yunlin 0.434-0.547 (county, all classes, 2016-2025)",
         "passed": c8["pass_1e-6"]},
        {"criterion": "T5 pre-origin == full on months <= 2022-12 (pumps, kept sets, cells)",
         "threshold": "max |d| <= 1e-9 kWh, identical NaN patterns and uid sets",
         "measured": "max |d| 0.0 kWh on 7 items; uid sets identical", "passed": t["T5"]["pass"]},
        {"criterion": "T6 masks and internal consistency (pre-install zeros never imputed, "
                      "2026-07 phase-A NaN/flag 8, NaN<=>flag 8, validate_grid, share x reading, "
                      "cells = sum of pumps, totals, long table, kept set recomputed)",
         "threshold": "0 violations; (iv) within 1e-6 kWh + 1e-9 rel",
         "measured": f"violations {t['T6']['violations']}; kept-set symmetric differences 0 "
                     "in all 9 comparisons", "passed": t["T6"]["pass"]},
        {"criterion": "T7 every county with pumps in active cells included",
         "threshold": "Yunlin, Changhua > 0; Chiayi 320; Pingtung 0",
         "measured": "Yunlin 114,059 / Changhua 113,113 kept pumps in grid; Chiayi 320 census "
                     "pumps in grid (316 kept); Pingtung 0", "passed": t["T7"]["pass"]},
        {"criterion": "T8 privacy: no personal column, no raw meter number in any output",
         "threshold": "0 violations",
         "measured": f"0 violations over {len(t['T8']['files_checked'])} files "
                     f"({t['T8']['n_raw_meter_numbers']} raw meter numbers scanned)",
         "passed": t["T8"]["pass"]},
        {"criterion": "T9 Changhua and Chiayi county axes equal scenario.energy_by_class on the "
                      "model's clean_census output (2012-2022)",
         "threshold": "every element |d| <= 1e-6 kWh + 1e-9 |ref|",
         "measured": "max |d| 0.0 kWh (both)", "passed": t["T9"]["pass"]},
    ]
    for k, g in s["gates"].items():
        acc.append({"criterion": f"gate {k}", "threshold": "conjunction of its registered tests",
                    "measured": ", ".join(f"{a}={b}" for a, b in g["tests"].items()),
                    "passed": g["passed"]})
    rz = lib.load_json(lib.EVIDENCE / "resume" / "resume_check.json")
    acc.append({"criterion": "resume integrity (2026-10-13, read-only, not a registered test): "
                             "manifests re-hash, pre-registration intact and still allowed, "
                             "independent spot check of the delivered arrays",
                "threshold": "0 hash failures; audit clean; spot identities <= 1e-6 kWh",
                "measured": f"{rz['files_rehashed']} files re-hashed ok={rz['manifests_ok']}; "
                            f"chains_intact={rz['prereg_audit']['chains_intact']}, C0 refused "
                            f"{len(rz['prereg_audit']['c0_refused'])}, changed inputs "
                            f"{len(rz['prereg_audit']['c0_changed_inputs'])}; spot "
                            f"{rz['spot_checks']}",
                "passed": bool(rz["manifests_ok"] and rz["prereg_status"]["allowed"]
                               and all(rz["spot_checks"].values()))})
    for crit, meas in DELIVERABLES:
        acc.append({"criterion": crit, "threshold": "delivered", "measured": meas,
                    "passed": True})
    st = dg["stats_preorigin"]
    out = {
        "wp": "C0", "status": "passed",
        "summary": SUMMARY,
        "prereg": {"id": lib.PREREG_ID, "sha256": t["T1a"]["prereg_sha256"],
                   "created": "2026-10-12T16:11:55.928102Z"},
        "acceptance": acc,
        "products": PRODUCTS,
        "code_paths": [str(lib.CODE_DIR / f) for f in CODE_FILES],
        "deviations_from_plan": DEVIATIONS,
        "issues_for_downstream": ISSUES,
        "resume": {"time_utc": rz["time_utc"], "evidence": lib.rel(lib.EVIDENCE / "resume"),
                   "files_rehashed": rz["files_rehashed"], "manifests_ok": rz["manifests_ok"],
                   "prereg_allowed": rz["prereg_status"]["allowed"],
                   "spot_checks": rz["spot_checks"], "ledger": rz["ledger"]},
        "key_numbers": {
            "in_grid_GWh_per_yr_2012_2022": {c: st[c]["in_grid_GWh_per_yr_2012_2022_all"]
                                             for c in ("yunlin", "changhua", "chiayi")},
            "domestic_share_in_grid_2012_2022": {c: st[c]["domestic_share_2012_2022"]
                                                 for c in ("yunlin", "changhua")},
            "seasonal_cv_2012_2022": {c: st[c]["seasonal_cv_2012_2022"]
                                      for c in ("yunlin", "changhua")},
            "coding_sensitivity": dg["coding_sensitivity_2012_2022"],
            "evensplit_effect": dg["stats_full"]["evensplit_vs_primary"],
            "t4": {k: t["T4"][k] for k in ("annual_coverage", "changhua_over_yunlin",
                                            "month_of_year_coverage_2016_2022_county_all",
                                            "monthly_correlation_census_county_all_vs_tpc")},
        },
        "evidence": lib.rel(lib.EVIDENCE),
        "claude_api_tokens": 0, "gpu": "none",
    }
    lib.save_json(out, lib.DATAPREP / "reports" / "C0.json")
    print("ok", all(a["passed"] for a in acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

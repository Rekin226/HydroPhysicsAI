"""WP C0: after the registered checks, write each product's validation.json, PRODUCT_CARD.md and
MANIFEST.json (library ``common.manifest``), and a post-finalize privacy re-scan of the files
added here.

Usage: python -I finalize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402

CHECKS = lib.EVIDENCE / "checks"
CODE = [lib.CODE_DIR / f for f in ("lib.py", "build.py", "assemble.py", "checks.py",
                                    "finalize.py")] + [
    lib.REPO / "hydrophysics/twin/pumping.py", lib.REPO / "hydrophysics/twin/scenario.py"]
L0 = [lib.PARQ / f"{c}_{k}.parquet" for c in lib.COUNTIES for k in ("census", "kwh_long")]
VALIDATION_INPUTS = [lib.PARQ / "pingtung_census.parquet", lib.AMP_CENSUS, lib.AMP_KWH,
                     lib.MODEL_INPUTS_PKL, lib.P1_CENSUS_CHECK / "census_truth_monthly.parquet",
                     lib.P1_CENSUS_CHECK / "evaluation.json",
                     lib.P1_TPC_OBSERVED / "tpc_county_sector_monthly_observed.parquet"]
ALL_USES = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
            "calibration_target", "initial_conditions", "idw_field", "prior",
            "prior_width_input", "model_design", "selection", "forcing_frozen",
            "forcing_recorded_arm", "validator", "statistical_layer_history",
            "after_the_fact_analysis", "design_time_evidence"]
ALLOWED = {
    "census_L1": ["model_design", "prior", "prior_width_input", "design_time_evidence",
                  "validator", "after_the_fact_analysis"],
    "preorigin": ["forcing_frozen", "model_design", "prior", "prior_width_input", "selection",
                  "design_time_evidence", "validator", "scoring_target",
                  "after_the_fact_analysis"],
    "full": ["forcing_recorded_arm", "statistical_layer_history", "validator",
             "after_the_fact_analysis"],
}
FWE = {"census_L1": "2022-12", "preorigin": "2022-12", "full": "2026-07"}
LAYER = {"census_L1": "L1", "preorigin": "L2", "full": "L2"}
PERSONAL = {
    "census_L1": "pump coordinates (TWD97) and well numbers (WELL_NO in sid/uid); no names, "
                 "addresses, postal codes or raw meter numbers (plan 2.6); gitignored",
    "preorigin": "well numbers (uid) in the per-pump arrays; no coordinates, names, addresses, "
                 "postal codes or raw meter numbers; gitignored",
    "full": "well numbers (uid) in the per-pump arrays; no coordinates, names, addresses, "
            "postal codes or raw meter numbers; gitignored",
}
CENSUS_NOTE = ("census attributes (pump list, HP, purpose, coordinates, install date, billing "
               "code) are the 2026 delivery; for Yunlin they equal the AMP_V2 snapshot the "
               "model's forcing used; installs are frozen at mid-2016, so no post-origin pump "
               "enters. No kWh reading after 2022-12 enters the pre-origin variant or any "
               "fitted quantity of the L1 table.")


def card(name: str, d: Path, summ: dict, gate: dict, build_info: dict | None) -> str:
    g = "PASSED" if gate["passed"] else "FAILED"
    lines = [f"# pump_electricity/{name}/{d.name} (WP C0)", "",
             f"Gate: **{g}** ({gate['criteria']}).", ""]
    if name == "census_L1":
        lines += [
            "One row per census pump: Yunlin and Changhua (whole counties), Chiayi pumps in "
            "the model's active cells plus their meter-mates. Harmonised fields: county, "
            "row_idx (L0), sid/uid (WELL_NO; Chiayi 'r<row_idx>'), town, work_station, "
            "irrigation_district, x/y (TWD97), grid row/col, cell (active index, -1 outside), "
            "pump_hp (+ raw string), purpose_raw / purpose_class (scenario.purpose_class), "
            "billing_code (M0/M1/M2/M9), install month (ROC date parsed; month 00 read as "
            "January), tube diameter / well depth where the sheet has them, meter_idx "
            "(county-scoped integer; the raw meter number is never written), meter_n_pumps, "
            "meter_first_pump, meter_hp_sum, meter_duty_2012_2022, kept (clean_census cap 1.0 "
            "on 2012-2022), hp_share (of the kept meter).", "",
            f"Personal data: {PERSONAL[name]}.", ""]
    else:
        lines += [
            "Cleaned pump electricity in the twin's model units (kWh of the de-duplicated, "
            "duty-capped census, `pumping.clean_census` cap 1.0, duty on 2012-01..2022-12), "
            "per pump and per 1 km active cell x purpose class, monthly "
            + ("2007-01..2022-12." if name == "preorigin" else
               "2007-01..2026-07 (2026-07 masked)."), "",
            "Files: `cells_by_class.npz` (value/flag/sd (C, A, T) summed over counties; "
            "value_by_county/flag_by_county (K, C, A, T) for yunlin, changhua, chiayi; "
            "hp_installed, hp_missing, n_terminated), `cells_total.npz` (all classes and the "
            "domestic-excluded sensitivity variant, (A, T))"
            + (", `cells_by_class_evensplit.npz` and `pumps_*_evensplit.npz` (alternative: a "
               "bimonthly bill posted wholly to the second month re-split evenly, flag 1)"
               if name == "full" else "")
            + ", `pumps_<county>.npz` (per-pump value/flag/method (P, T), hp, hp_share, cell, "
            "class, meter_idx), `meters_<county>.parquet` (billing phase from the code and "
            "detected, install month, terminated meters), `county_class_monthly.parquet` "
            "(county and in-grid scopes, classes, all, all_nodomestic), `build_info.json`.", "",
            "Flags: 0 observed (method separates observed, pre_install structural zero, "
            "observed_before_install_date, terminated_trailing_run, bill_in_second_month, "
            "uneven_pair); 1 even-split alternative only; 8 missing, never filled (a "
            "post-install NaN reading; any aggregate containing a phase-A (M2) meter's "
            "2026-07, whose July-August bill is not issued); 9 suspect (aggregate whose missing "
            "readings exceed 10 % of its installed kept HP; monthly meters' 2026-07). Missing "
            "pump readings count as zero in aggregates, as in the model. sd is NaN (observed "
            "values).", ""]
        if build_info:
            rows = ["| county | kept pumps | kept in grid | kept meters | terminated meters "
                    "(in grid) |", "|---|---|---|---|---|"]
            for t, i in build_info["counties"].items():
                rows.append(f"| {t} | {i['pumps_kept']} | {i['pumps_kept_in_grid']} | "
                            f"{i['meters_kept']} | {i['meters_terminated']} "
                            f"({i['meters_terminated_in_grid']}) |")
            lines += rows + ["", "Chiayi: only the in-grid pumps are written per pump; its "
                             "county-scope rows of county_class_monthly cover all kept Chiayi "
                             "pumps.", ""]
    lines += ["Known limits:", "",
              "- The census is frozen at mid-2016 installs (no post-2016 pumps), as for "
              "Yunlin in the model.",
              "- 86-94 % of meters are billed bimonthly and TPC splits each bill evenly over "
              "its two months (a phase-locked two-month boxcar at cell scale); the P2a "
              "Chow-Lin split is not applied here.",
              "- From 2025-07 on, 12-33 % of bimonthly pairs post the whole bill to the second "
              "month (full variant; even-split alternative provided).",
              "- Changhua's domestic class (about 23,000 pumps) may carry household load; use "
              "the domestic-excluded variant as the sensitivity member.",
              "- Pre-install coding: the duty statistic sees the L0 readings as delivered "
              "(Yunlin pre-install zeros enter its mean, Changhua/Chiayi NaN do not); the "
              "alternative is quantified in build_info / the C0 report.",
              "- A phase-B (M1) meter's December value is half of a December-January bill, so "
              "the pre-origin 2022-12 carries half a month of January 2023 electricity for "
              "those meters (the model's own convention).",
              f"- {CENSUS_NOTE}",
              "- Coverage of TPC 農林漁牧 sales is about 0.4 and not stationary (descriptive; "
              "see the C0 report, T4).", "",
              f"Allowed uses: {', '.join(ALLOWED[name])}. "
              + ("Pre-origin uses (forcing of calibration and frozen forecasts) take this "
                 "variant." if name == "preorigin" else
                 ("Full variant: recorded-forcing diagnostic arm only (U4)."
                  if name == "full" else "")), "",
              f"Pre-registration: {lib.PREREG_ID} (sha256 {summ.get('prereg_sha256', '')}); "
              "acceptance in validation.json; WP report results/dataprep/reports/C0.md.", ""]
    return "\n".join(lines)


def main() -> int:
    prods = {k: lib.REPO / v for k, v in lib.load_json(lib.WORK / "products.json").items()}
    summ = lib.load_json(CHECKS / "summary.json")
    tests = {k: lib.load_json(CHECKS / f"{k}.json")
             for k in ("T1a", "T1b", "T2a", "T2b", "T3", "T4", "T5", "T6", "T7", "T8", "T9")
             if (CHECKS / f"{k}.json").exists()}
    sha = next((t.get("prereg_sha256") for t in tests.values() if t.get("prereg_sha256")), "")
    summ["prereg_sha256"] = sha
    gate_key = {"census_L1": "pump_electricity/census_L1",
                "preorigin": "pump_electricity/preorigin", "full": "pump_electricity/full"}
    written = {}
    for name, d in prods.items():
        g = summ["gates"][gate_key[name]]
        gate = {"passed": bool(g["passed"]),
                "criteria": f"pre-registered {lib.PREREG_ID}: conjunction of "
                            + ", ".join(f"{k}={v}" for k, v in g["tests"].items()),
                "tests": g["tests"]}
        val = {"prereg": lib.PREREG_ID, "prereg_sha256": sha, "gate": gate,
               "reported_only": summ["reported_only"],
               "evidence_dir": lib.rel(CHECKS),
               "evidence_sha256": {f"{k}.json": sha256_file(CHECKS / f"{k}.json")
                                   for k in tests},
               "tests": {k: {kk: vv for kk, vv in t.items()
                             if kk not in ("post_registration_looks", "files_checked")}
                         for k, t in tests.items()
                         if k in ({"T6", "T7", "T8"} if name == "census_L1" else set(tests))}}
        lib.save_json(val, d / "validation.json")
        bi = lib.load_json(d / "build_info.json") if name != "census_L1" else None
        (d / "PRODUCT_CARD.md").write_text(card(name, d, summ, gate, bi), encoding="utf-8")
        params = {"prereg": {"id": lib.PREREG_ID, "sha256": sha},
                  "cleaning": "pumping.clean_census cap_duty 1.0, t0 2012-01-01, t1 "
                              "2023-01-01 (duty statistic); kept pumps' record re-shared "
                              "with cap None (HP split of de-duplicated meters)",
                  "preinstall_coding": "duty statistic over the L0 readings as delivered; "
                                       "pre-install months are structural zeros in the product",
                  "classes": list(lib.CLASSES), "counties": list(lib.COUNTIES),
                  "grid": lib.grid_meta(lib.load_grid()),
                  "flag_9_hp_missing_frac": lib.HP_MISSING_FLAG_FRAC,
                  "termination_min_run_months": lib.TERMINATION_MIN_RUN,
                  "pair_identity_atol_kwh": lib.PAIR_ATOL,
                  "variant": lib.VARIANTS.get(name),
                  "census_snapshot_note": CENSUS_NOTE,
                  "validation_inputs": [{"path": lib.rel(p), "sha256": sha256_file(p)}
                                        for p in VALIDATION_INPUTS],
                  "census_licence": "restricted; local research use only; do not "
                                    "redistribute (SOURCE.json)"}
        inputs = [lib.rel(p) for p in L0] + [lib.rel(lib.SOURCE_JSON), lib.rel(lib.GRID_NPZ)]
        m = manifest.build_manifest(
            d, product=f"pump_electricity/{name}", layer=LAYER[name], inputs=inputs,
            code=[lib.rel(p) for p in CODE], parameters=params, fit_window_end=FWE[name],
            validation_report="validation.json", gate=gate, allowed_uses=ALLOWED[name],
            forbidden_uses=[u for u in ALL_USES if u not in ALLOWED[name]],
            personal_data=PERSONAL[name])
        errs = manifest.validate_manifest(m, d)
        if errs:
            raise SystemExit(f"{name}: manifest errors {errs}")
        manifest.write_manifest(d, m)
        written[name] = lib.rel(d / "MANIFEST.json")
    # privacy re-scan of the files written here (names only; T8 covered the rest)
    bad = []
    forbidden = list(lib.FORBIDDEN_COLS) + ["電號1", "電號2"]
    for d in prods.values():
        for f in ("validation.json", "PRODUCT_CARD.md", "MANIFEST.json"):
            txt = (d / f).read_text(encoding="utf-8")
            bad += [f"{d.name}/{f}:{n}" for n in forbidden if n in txt]
    lib.save_json({"manifests": written, "privacy_rescan_hits": bad},
                  lib.EVIDENCE / "finalize.json")
    print(written, "privacy hits:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

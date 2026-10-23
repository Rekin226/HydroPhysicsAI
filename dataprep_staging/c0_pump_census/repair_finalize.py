"""WP C0 repair round: after the registered checks, write validation.json, PRODUCT_CARD.md and
MANIFEST.json of the three v2 products (``common.manifest``), then a privacy re-scan of the
files written here.

The census attribute table (2026 delivery, data to 2026-07) is declared as an exemption of the
pre-origin and L1 products (plan 2.3, U3): fit_window_end 2026-07, non_exempt_fit_window_end
2022-12, the exemptable pre-origin uses named explicitly, ledger row and sensitivity run (R9).

Usage: python -I repair_finalize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair_lib as rl  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.uses import EXEMPTABLE_USES  # noqa: E402

CHECKS = lib.EVIDENCE / "repair" / "checks"
PREREG_ID = "C0.census_product.v2"
CODE = [lib.CODE_DIR / f for f in ("lib.py", "build.py", "checks.py", "repair_lib.py",
                                    "repair_build.py", "repair_assemble.py",
                                    "repair_checks.py", "repair_finalize.py")] + [
    lib.REPO / "hydrophysics/twin/pumping.py", lib.REPO / "hydrophysics/twin/scenario.py"]
L0 = [lib.PARQ / f"{c}_{k}.parquet" for c in lib.COUNTIES for k in ("census", "kwh_long")]
V1 = lib.PRODUCT_ROOT
VALIDATION_REFS = [lib.PARQ / "pingtung_census.parquet", lib.AMP_CENSUS, lib.AMP_KWH,
                   lib.MODEL_INPUTS_PKL, lib.EVIDENCE / "checks" / "T2a.json"] + [
    V1 / v / "v1" / f for v in ("preorigin", "full")
    for f in ("pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz",
              "cells_by_class.npz")] + [V1 / "census_L1" / "v1" / "census_pumps.parquet"]
ALL_USES = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
            "calibration_target", "initial_conditions", "idw_field", "prior",
            "prior_width_input", "model_design", "selection", "forcing_frozen",
            "forcing_recorded_arm", "validator", "statistical_layer_history",
            "after_the_fact_analysis", "design_time_evidence"]
ALLOWED = {
    "census_L1": ["model_design", "prior", "prior_width_input", "design_time_evidence",
                  "validator", "after_the_fact_analysis"],
    "preorigin": ["forcing_frozen", "model_design", "prior", "prior_width_input",
                  "design_time_evidence", "validator", "after_the_fact_analysis"],
    "full": ["forcing_recorded_arm", "statistical_layer_history", "validator",
             "after_the_fact_analysis"],
}
LAYER = {"census_L1": "L1", "preorigin": "L2", "full": "L2"}
PERSONAL = {
    "census_L1": "pump coordinates (TWD97) and well numbers (WELL_NO in sid/uid); no names, "
                 "addresses, postal codes or raw meter numbers (plan 2.6); gitignored",
    "preorigin": "well numbers (uid) in the per-pump arrays; no coordinates, names, addresses, "
                 "postal codes or raw meter numbers; gitignored",
    "full": "well numbers (uid) in the per-pump arrays; no coordinates, names, addresses, "
            "postal codes or raw meter numbers; gitignored",
}


def exemption(name: str, r9: dict, checks_dir: Path = CHECKS) -> dict:
    pc = r9["per_county"]
    sens = "; ".join(
        f"{t}: {pc[t]['pumps_install_after_2022_12']} pumps installed after 2022-12 "
        f"(latest {pc[t]['latest_valid_install']}), "
        + (f"in-grid 2012-2022 energy {pc[t]['rel_change_2012_2022']:+.2e} relative and max "
           f"monthly {pc[t]['max_rel_month_2012_2022']:.2e} without them"
           if "rel_change_2012_2022" in pc[t] else "no rebuild needed")
        for t in lib.COUNTIES)
    return {
        "input": "census attribute table of the TPC 2026 delivery (parquet/<county>_census."
                 "parquet: pump list, HP, purpose, coordinates, install dates, billing codes)",
        "data_end": "2026-07",
        "reason": "no attribute snapshot as of 2022-12 exists for Changhua or Chiayi (the AMP_V2 "
                  "Yunlin snapshot used by the calibrated model is itself post-origin and "
                  "equals the 2026 sheet); install dates show the table includes pumps "
                  "installed after the origin",
        "ledger_number_id": f"C0.v2.exemption.census_attributes.{name}",
        "sensitivity": f"pre-registered R9 ({lib.rel(checks_dir / 'R9.json')}): {sens}",
        "uses": [u for u in ALLOWED[name] if u in EXEMPTABLE_USES],
    }


def card(name: str, d: Path, gate: dict, ev: dict, supersedes: str = "v1",
         prereg_id: str = PREREG_ID) -> str:
    g = "PASSED" if gate["passed"] else "FAILED"
    r7, r8, r10 = ev.get("R7", {}), ev.get("R8", {}), ev.get("R10", {})
    sdl = r7.get("sd_label", "uncalibrated")
    lines = [f"# pump_electricity/{name}/{d.name} (WP C0, repair round)", "",
             f"Gate: **{g}** ({gate['criteria']}).", ""]
    if not gate["passed"]:
        lines += ["**This version's registered gate failed (see validation.json and the C0 "
                  "report, Repair round): do not use it; use the next version of this "
                  "product.**", ""]
    lines += [f"Supersedes `pump_electricity/{name}/{supersedes}`. The v1 pre-origin variant carried "
             "January-2023 electricity (the 2022-12 value of every Dec-Jan-paired meter is "
             "half of a December-2022/January-2023 bill, and the duty statistic saw it); v1 "
             "must not be used for pre-origin purposes. v2 removes that information everywhere "
             "(registered invariance test R2).", ""]
    if name == "census_L1":
        lines += [
            "One row per census pump (Yunlin, Changhua; Chiayi pumps in active cells plus "
            "their meter-mates), harmonised fields as v1, with the pre-origin kept set and "
            "duty: `meter_duty_2012_2022` is the mean 2012-2022 draw without the 2022-12 value "
            "of Dec-Jan-paired meters; `kept` uses it (cap 1.0); `dec_jan_paired` and "
            "`phase_detected_2012_2022` (identical-pair detection on pairs inside "
            "2012-01..2022-11) are new.", "",
            f"Personal data: {PERSONAL[name]}.", ""]
    else:
        pre = name == "preorigin"
        lines += [
            "Cleaned pump electricity in the twin's model units (kWh of the de-duplicated, "
            "duty-capped census, `pumping.clean_census` cap 1.0), per pump and per 1 km active "
            "cell x purpose class, monthly "
            + ("2007-01..2022-12." if pre else "2007-01..2026-07 (2026-07 masked)."), "",
            "Kept set (both variants): duty over 2012-2022 of the pre-origin readings, without "
            "the 2022-12 value of Dec-Jan-paired meters, so both variants share it and the "
            "recorded-forcing arm differs from the frozen arm only in post-origin forcing.", ""]
        if pre:
            lines += [
                "**December 2022 of Dec-Jan-paired meters** (identical-pair phase B on "
                "2012-01..2022-11, or code M1 with an inconclusive detection): flag 4, method "
                "`fill_dec2022_dec_jan_pair`, value = rho_bar(county, class) x (Oct-2022 + "
                "Nov-2022)/2, the ratio of Dec_y+Jan_y+1 to Oct_y+Nov_y averaged over 2012-2021 "
                "(`fill_parameters.json`). It estimates the convention value of every other "
                "December of these meters (the even-split half of the Dec-Jan bill) from "
                "data read by 2022-12. Without an Oct-Nov 2022 base the value is missing "
                "(flag 8). sd: common ratio factor (fully correlated) plus a meter term; "
                f"**{sdl}** by the pre-registered leave-one-year-out back-test R7 "
                + (f"(class x cell coverage of the 90 % interval "
                   f"{r7['class_x_cell']['coverage_90']:.3f}, Yunlin+Changhua county "
                   f"{r7['county_in_grid']['coverage_90_yunlin_changhua']:.3f}; county median "
                   f"|error| {r7['county_in_grid']['median_abs_rel_error']:.3f})"
                   if r7 else "") + ".", ""]
            if supersedes != "v1":
                lines += ["v3 change from v2: the Oct-Nov 2022 base is coded as the product codes "
                          "those months (pre-install NaN or 0 = structural zero; positive "
                          "readings before the census install date kept), so a meter operating "
                          "before its recorded install date is filled; a meter not yet installed "
                          "in 2022-12 whose base is 0 stays a structural zero by date.", ""]
            if r8:
                pc = r8["per_county"]
                lines += ["After the fact (R8, January-2023 data, never fed back): in-grid fill "
                          "vs the v1 value of the same pumps " + ", ".join(
                              f"{t} {pc[t]['fill_minus_actual_rel']:+.1%}" for t in pc
                              if pc[t]['fill_minus_actual_rel'] is not None) + ".", ""]
        lines += [
            "Files: `cells_by_class.npz` (value/flag/sd (C, A, T); value_by_county/"
            "flag_by_county/sd_by_county (K, C, A, T); hp_installed, hp_missing, "
            "n_terminated, n_fill, missing_expected_kwh), `cells_total.npz` (all classes and "
            "the domestic-excluded sensitivity variant with flags, sd and "
            "missing_expected_kwh)"
            + (", `fill_parameters.json`" if pre else
               ", `cells_by_class_evensplit.npz` and `pumps_*_evensplit.npz` (a bimonthly "
               "bill posted wholly to the second month re-split evenly, flag 1)")
            + ", `pumps_<county>.npz` (per-pump value/flag/method"
            + ("/sd" if pre else "") + ", hp, hp_share, cell, class, meter_idx), "
            "`meters_<county>.parquet` (billing phase from the code and detected, "
            "dec_jan_paired, pre-origin duty, install month, termination"
            + (", fill base/value/sd" if pre else "") + "), `county_class_monthly.parquet` "
            "(county and in-grid scopes, classes, all, all_nodomestic; value, sd, flag, "
            "method, hp_installed, hp_missing, missing_expected_kwh, n_fill), "
            "`build_info.json`.", "",
            "Flags: 0 observed (method separates observed, pre_install structural zero, "
            "observed_before_install_date, terminated_trailing_run, bill_in_second_month, "
            "uneven_pair); "
            + ("4 the December-2022 fill (pumps) or an aggregate holding it; " if pre else
               "1 even-split alternative only; ")
            + "8 missing, never filled; 9 suspect (aggregate whose missing readings exceed "
            "10 % of its installed kept HP"
            + ("" if pre else "; monthly meters' 2026-07") + "). Aggregate precedence 8 > 9 > "
            "4 > 0. Missing pump readings count as zero in aggregates, as in the model; "
            "`missing_expected_kwh` gives their expected kWh (HP x same-month county-class "
            "mean kWh/HP of reporting pumps; an indicator, not a fill).", ""]
        if r10:
            v = r10.get(name, {})
            if v:
                lines += ["In-grid GWh/yr 2012-2022: " + ", ".join(
                    f"{t} {v[t]['in_grid_GWh_per_yr_2012_2022']:.2f} (domestic "
                    f"{v[t]['domestic_share_2012_2022']:.1%})" for t in lib.COUNTIES) + ".", ""]
    lines += ["Known limits:", "",
              "- The census attribute table is the 2026 delivery: it includes pumps installed "
              "after the origin (declared exemption of the pre-origin and L1 products; R9 "
              "sensitivity).",
              "- 86-94 % of meters are billed bimonthly and TPC splits each bill evenly over its "
              "two months (flag 0, method observed): monthly values of bimonthly meters are "
              "even-split halves. No monthly-disaggregation test may score against them; this "
              "product does not allow scoring_target.",
              "- Changhua's domestic class may carry household load; use the domestic-excluded "
              "variant as the sensitivity member.",
              "- Pre-install coding as v1: the duty statistic sees the L0 readings as delivered "
              "(Yunlin pre-install zeros enter its mean, Changhua/Chiayi NaN do not).",
              "- The TPC reading dates are undocumented: a reading taken in the first days of "
              "January would add a few days of January to Nov-Dec pairs and monthly meters' "
              "December; not resolvable from the data.",
              "- Coverage of TPC 農林漁牧 sales is about 0.4 and not stationary (descriptive; v1 "
              "T4).", "",
              f"Allowed uses: {', '.join(ALLOWED[name])}."
              + (" Pre-origin uses take this variant (through the census-attribute exemption)."
                 if name == "preorigin" else
                 (" Recorded-forcing diagnostic arm only (U4)." if name == "full" else "")), "",
              f"Pre-registration: {prereg_id}; acceptance in validation.json; WP report "
              "results/dataprep/reports/C0.md (Repair round).", ""]
    return "\n".join(lines)


def main(prods_json: Path = lib.EVIDENCE / "repair" / "products_v2.json",
         checks_dir: Path = CHECKS, prereg_id: str = PREREG_ID, code: list | None = None,
         supersedes: str = "v1", out_json: Path = lib.EVIDENCE / "repair" / "finalize.json"
         ) -> int:
    CHECKS = checks_dir  # noqa: N806
    PREREG_ID = prereg_id  # noqa: N806
    CODE_ = code or CODE
    prods = {k: lib.REPO / v for k, v in lib.load_json(prods_json).items()}
    summ = lib.load_json(CHECKS / "summary.json")
    keys = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "T5")
    ev = {k: lib.load_json(CHECKS / f"{k}.json") for k in keys if (CHECKS / f"{k}.json").exists()}
    sha = next((t.get("prereg_sha256") for t in ev.values() if t.get("prereg_sha256")), "")
    gate_key = {k: f"pump_electricity/{k}" for k in prods}
    written = {}
    for name, d in prods.items():
        g = summ["gates"][gate_key[name]]
        gate = {"passed": bool(g["passed"]),
                "criteria": f"pre-registered {PREREG_ID}: conjunction of "
                            + ", ".join(f"{k}={v}" for k, v in g["tests"].items()),
                "tests": g["tests"]}
        keep = {"census_L1": {"R2", "R4", "R5", "R6", "R9"},
                "preorigin": set(keys), "full": {"R1", "R4", "R5", "R6", "T5", "R10"}}[name]
        val = {"prereg": PREREG_ID, "prereg_sha256": sha, "gate": gate,
               "sd_label": summ.get("sd_label") if name == "preorigin" else None,
               "reported_only": summ["reported_only"], "evidence_dir": lib.rel(CHECKS),
               "evidence_sha256": {f"{k}.json": sha256_file(CHECKS / f"{k}.json") for k in ev},
               "tests": {k: {kk: vv for kk, vv in t.items()
                             if kk not in ("post_registration_looks", "files", "table")}
                         for k, t in ev.items() if k in keep},
               "supersedes": f"pump_electricity/{name}/{supersedes}"}
        lib.save_json(val, d / "validation.json")
        (d / "PRODUCT_CARD.md").write_text(card(name, d, gate, ev, supersedes, PREREG_ID),
                                           encoding="utf-8")
        params = {"prereg": {"id": PREREG_ID, "sha256": sha},
                  "cleaning": "pumping.clean_census cap_duty 1.0, duty mean over 2012-01..2022-12 "
                              "of the L0 readings as delivered without the 2022-12 value of "
                              "Dec-Jan-paired meters; kept pumps' record re-shared with cap None",
                  "dec_jan_paired_rule": "identical-pair phase detected B on pairs inside "
                                         "2012-01..2022-11, or billing code M1 with detection "
                                         "in (mixed, insufficient, both-flat)",
                  "dec2022_fill": ("rho_bar(county, class) x (Oct+Nov 2022)/2, flag 4; see "
                                   "fill_parameters.json" if name == "preorigin" else
                                   "none (full variant keeps the recorded 2022-12 values)"),
                  "fill_fit_years": list(rl.FIT_YEARS), "n_min_class": rl.N_MIN_CLASS,
                  "classes": list(lib.CLASSES), "counties": list(lib.COUNTIES),
                  "grid": lib.grid_meta(lib.load_grid()),
                  "flag_9_hp_missing_frac": lib.HP_MISSING_FLAG_FRAC,
                  "termination_min_run_months": lib.TERMINATION_MIN_RUN,
                  "pair_identity_atol_kwh": lib.PAIR_ATOL,
                  "variant": lib.VARIANTS.get(name),
                  "census_licence": "restricted; local research use only; do not "
                                    "redistribute (SOURCE.json)",
                  "supersedes": f"pump_electricity/{name}/{supersedes}"}
        inputs = ([lib.rel(p) for p in L0] + [lib.rel(lib.SOURCE_JSON), lib.rel(lib.GRID_NPZ)]
                  + [{"path": lib.rel(p), "role": "validation_reference: read only by the "
                      "acceptance checks (repair_checks.py), never by the build"}
                     for p in VALIDATION_REFS])
        ex, ne, fwe = [], None, "2026-07"
        if name in ("preorigin", "census_L1"):
            ex = [exemption(name, ev["R9"], CHECKS)]
            ne = "2022-12"
        m = manifest.build_manifest(
            d, product=f"pump_electricity/{name}", layer=LAYER[name], inputs=inputs,
            code=[lib.rel(p) for p in CODE_], parameters=params, fit_window_end=fwe,
            validation_report="validation.json", gate=gate, allowed_uses=ALLOWED[name],
            forbidden_uses=[u for u in ALL_USES if u not in ALLOWED[name]],
            personal_data=PERSONAL[name], exemptions=ex, non_exempt_fit_window_end=ne)
        errs = manifest.validate_manifest(m, d)
        if errs:
            raise SystemExit(f"{name}: manifest errors {errs}")
        manifest.write_manifest(d, m)
        written[name] = lib.rel(d / "MANIFEST.json")
    bad = []
    forbidden = list(lib.FORBIDDEN_COLS) + ["電號1", "電號2"]
    for d in prods.values():
        for f in ("validation.json", "PRODUCT_CARD.md", "MANIFEST.json"):
            txt = (d / f).read_text(encoding="utf-8")
            bad += [f"{d.name}/{f}:{n}" for n in forbidden if n in txt]
    lib.save_json({"manifests": written, "privacy_rescan_hits": bad}, out_json)
    print(written, "privacy hits:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

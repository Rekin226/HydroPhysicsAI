"""WP C0 repair round, v3: write validation.json, PRODUCT_CARD.md and MANIFEST.json of the v3
products (``common.manifest``) from the registered v3 checks, then a privacy re-scan of the
files written here.

Rules as ``repair_finalize`` (v2), whose constants are reused, with the v3 differences:

* two exemptions on the pre-origin and L1 products (plan 2.3, U3), each with a ledger id and a
  sensitivity run:
  - the census attribute table of the 2026 delivery (data to 2026-07; sensitivity: registered
    R9 of C0.census_product.v3);
  - the residual post-origin information: the 2022-12 readings of meters that the pre-origin
    rule cannot label Dec-Jan-paired but whose December-2022 reading equals January-2023 (data
    to 2023-01; sensitivity: registered R8, after the fact, and ``repair3_residual.py``);
* card and validation texts for v3 (supersedes v2, whose registered gate failed, and v1);
* validation references also hash the v2 product files R1/R3 read and the design look.

Usage: python -I repair3_finalize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair_finalize as rf  # noqa: E402  (constants of the v2 finaliser, unchanged)
import repair_lib as rl  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.uses import EXEMPTABLE_USES  # noqa: E402

EV3 = lib.EVIDENCE / "repair3"
CHECKS = EV3 / "checks"
PREREG_ID = "C0.census_product.v3"
RESIDUAL = EV3 / "residual_sensitivity.json"
RESUME = EV3 / "resume_check.json"
FACTS = EV3 / "report_facts.json"
PE = lib.PRODUCT_ROOT
CODE = [lib.CODE_DIR / f for f in (
    "lib.py", "build.py", "checks.py", "repair_lib.py", "repair_build.py", "repair_assemble.py",
    "repair_finalize.py", "repair3_lib.py", "repair3_build.py", "repair3_assemble.py",
    "repair3_checks.py", "repair3_residual.py", "repair3_facts.py", "repair3_finalize.py")] + [
    lib.REPO / "hydrophysics/twin/pumping.py", lib.REPO / "hydrophysics/twin/scenario.py"]
V2_REFS = [PE / "preorigin" / "v2" / f for f in (
    "pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz", "cells_by_class.npz")]
DESIGN_LOOK = lib.EVIDENCE / "repair" / "design_look.json"


def _f(x: float, n: int = 0) -> str:
    return f"{x:,.{n}f}"


def exemption_census(name: str, r9: dict) -> dict:
    pc = r9["per_county"]
    sens = "; ".join(
        f"{t}: {pc[t]['pumps_install_after_2022_12']} pumps installed after 2022-12 "
        f"(latest {pc[t]['latest_valid_install']}), in-grid 2012-2022 energy "
        f"{pc[t]['rel_change_2012_2022']:+.2e} relative and max monthly "
        f"{pc[t]['max_rel_month_2012_2022']:.2e} without them" for t in lib.COUNTIES)
    return {
        "input": "census attribute table of the TPC 2026 delivery (parquet/<county>_census."
                 "parquet: pump list, HP, purpose, coordinates, install dates, billing codes)",
        "data_end": "2026-07",
        "reason": "no attribute snapshot as of 2022-12 exists for Changhua or Chiayi (the AMP_V2 "
                  "Yunlin snapshot used by the calibrated model is itself post-origin and "
                  "equals the 2026 sheet); install dates show the table includes pumps "
                  "installed after the origin",
        "ledger_number_id": f"C0.v3.exemption.census_attributes.{name}",
        "sensitivity": f"pre-registered R9 of {PREREG_ID} ({lib.rel(CHECKS / 'R9.json')}): "
                       f"{sens}",
        "uses": [u for u in rf.ALLOWED[name] if u in EXEMPTABLE_USES],
    }


def exemption_residual(name: str, res: dict) -> dict:
    pc = res["per_county"]
    sens = "; ".join(
        f"{t}: {pc[t]['untreated_meters_all']} meters ({pc[t]['untreated_meters_kept_in_grid']} "
        f"kept in grid), 2022-12 in-grid {_f(pc[t]['dec2022_in_grid_kWh_on_untreated_meters'], 1)}"
        f" kWh = {pc[t]['share_of_dec2022_in_grid']:.3%} of the in-grid Dec-2022 and "
        f"{pc[t]['share_of_2012_2022_in_grid']:.1e} of 2012-2022; kept flips "
        f"{pc[t]['kept_flips_kept_to_dropped'] + pc[t]['kept_flips_dropped_to_kept']}"
        for t in lib.COUNTIES)
    return {
        "input": "L0 kWh readings (parquet/<county>_kwh_long.parquet): the 2022-12 reading of "
                 "meters that the pre-origin rule does not label Dec-Jan-paired but whose "
                 "December-2022 and January-2023 readings are equal non-zero halves, i.e. "
                 "probably half of a December-2022/January-2023 bill read in 2023; kept as read "
                 "in the values and in the duty statistic",
        "data_end": "2023-01",
        "reason": "the meters cannot be identified without the January-2023 reading (their "
                  "2012-01..2022-11 history shows no Dec-Jan pairing); a pre-origin rule can "
                  "neither fill nor mask them, and selecting them with the January-2023 "
                  "reading would itself use post-origin data",
        "ledger_number_id": f"C0.v3.exemption.residual_unpaired_dec2022.{name}",
        "sensitivity": f"registered R8 of {PREREG_ID} (after the fact) and repair3_residual.py "
                       f"({lib.rel(RESIDUAL)}): {sens}",
        "uses": [u for u in rf.ALLOWED[name] if u in EXEMPTABLE_USES],
    }


def card(name: str, d: Path, gate: dict, ev: dict, res: dict, facts: dict,
         binfo: dict) -> str:
    g = "PASSED" if gate["passed"] else "FAILED"
    r1, r3, r7, r8, r10 = ev["R1"], ev["R3"], ev["R7"], ev["R8"], ev["R10"]
    pre = name == "preorigin"
    pc = res["per_county"]
    L = [f"# pump_electricity/{name}/{d.name} (WP C0, repair round)", "",
         f"Gate: **{g}** ({gate['criteria']}).", ""]
    if not gate["passed"]:
        L += ["**This version's registered gate failed: do not use it.**", ""]
    L += [f"Supersedes `pump_electricity/{name}/v2` (its registered gate failed: an R4 check "
          "defect and a T5 product defect, fixed here) and `pump_electricity/" + name + "/v1`. "
          "The v1 pre-origin variant carried January-2023 electricity: the 2022-12 value of "
          "every Dec-Jan-paired (phase-B) meter is half of a December-2022/January-2023 bill "
          "read in 2023 (about 15 % of the in-grid December-2022 forcing), and its duty "
          "statistic saw it. **Do not use v1 or v2 for any pre-origin purpose.** Their "
          "manifests cannot be changed, and v1's still lists forcing_frozen. v3 removes that "
          "information. In the registered invariance test R2, every month >= 2023-01 and the "
          "2022-12 reading of every paired meter were perturbed: the v3 pre-origin products "
          "and the L1 columns stayed identical, while v1's builder changed on the same "
          "input.", ""]
    flips = ", ".join(f"{t} +{r1['per_county'][t]['flips_in_meters']}/-"
                      f"{r1['per_county'][t]['flips_out_meters']}" for t in lib.COUNTIES)
    if name == "census_L1":
        L += ["One row per census pump (Yunlin, Changhua; Chiayi pumps in active cells plus "
              "their meter-mates), harmonised fields as v1, with the pre-origin kept set. "
              "`meter_duty_2012_2022` is the mean 2012-2022 draw without the 2022-12 value of "
              "Dec-Jan-paired meters, and `kept` uses it (cap 1.0). `dec_jan_paired` and "
              "`phase_detected_2012_2022` are new (identical-pair detection on pairs inside "
              f"2012-01..2022-11). Kept-meter changes against v1: {flips}.", ""]
    else:
        L += ["Cleaned pump electricity in the twin's model units: kWh of the de-duplicated, "
              "duty-capped census (`pumping.clean_census` cap 1.0), per pump and per 1 km "
              "active cell x purpose class, monthly "
              + ("2007-01..2022-12." if pre else "2007-01..2026-07 (2026-07 masked)."), "",
              "Kept set (shared by both variants): duty over 2012-2022 of the pre-origin "
              "readings, without the 2022-12 value of Dec-Jan-paired meters. The "
              "recorded-forcing arm therefore differs from the frozen arm only in post-origin "
              f"forcing. Kept-meter changes against v1: {flips}.", ""]
        if pre:
            cc, ci = r7["class_x_cell"], r7["county_in_grid"]
            cy_rows = [r for r in ci["table"] if r["county"] == "chiayi"]
            cy_over = sum(r["F"] > r["A"] for r in cy_rows)
            L += ["**December 2022 of Dec-Jan-paired meters** (identical-pair phase B on "
                  "2012-01..2022-11, or code M1 with an inconclusive detection) carries flag 4 "
                  "and method `fill_dec2022_dec_jan_pair`. The value is rho_bar(county, class) "
                  "x base, where base = (Oct-2022 + Nov-2022)/2 and rho_bar is the ratio "
                  "(Dec_y + Jan_y+1) / (Oct_y + Nov_y) averaged over 2012-2021 "
                  "(`fill_parameters.json`). It estimates the December half of the Dec-Jan "
                  "bill from data read by 2022-12. The Oct-Nov 2022 base is coded as the "
                  "product codes those months: a pre-install NaN or 0 is a structural zero, "
                  "and a positive reading before the census install date is kept. A meter "
                  "whose installed base month is missing stays missing (flag 8). A meter not "
                  "yet installed in 2022-12 with a zero base stays a structural zero.", "",
                  f"The fill's sd combines a fully correlated ratio factor with a meter term. "
                  f"It is labelled **{r7['sd_label']}** by the pre-registered "
                  f"leave-one-year-out back-test R7 (2012-2021):",
                  f"- nominal 90 % interval coverage: class x cell {cc['coverage_90']:.3f} "
                  f"(n {cc['n']}), Yunlin+Changhua county totals "
                  f"{ci['coverage_90_yunlin_changhua']:.2f};",
                  f"- median |relative error|: class x cell {cc['median_abs_rel_error']:.3f}, "
                  f"county {ci['median_abs_rel_error']:.3f};",
                  f"- Chiayi's fill exceeded the realised half in {cy_over} of "
                  f"{len(cy_rows)} back-test years.", "",
                  "After the fact (R8, January-2023 data, never fed back), the in-grid fill "
                  "differs from the realised December half of the same pumps by "
                  + ", ".join(f"{t} {r8['per_county'][t]['fill_minus_actual_rel']:+.1%}"
                              for t in lib.COUNTIES) + ".", ""]
        L += ["Files:",
              "- `cells_by_class.npz`: value/flag/sd (C, A, T); "
              "value_by_county/flag_by_county/sd_by_county (K, C, A, T); hp_installed, "
              "hp_missing, n_terminated, n_fill, missing_expected_kwh.",
              "- `cells_total.npz`: all classes and the domestic-excluded sensitivity variant, "
              "with flags, sd and missing_expected_kwh.",
              "- `fill_parameters.json`." if pre else
              "- `cells_by_class_evensplit.npz` and `pumps_*_evensplit.npz`: a bimonthly bill "
              "posted wholly to the second month, re-split evenly (flag 1).",
              "- `pumps_<county>.npz`: per-pump value/flag/method" + ("/sd" if pre else "")
              + ", hp, hp_share, cell, class, meter_idx.",
              "- `meters_<county>.parquet`: billing phase (from the code and detected), "
              "dec_jan_paired, pre-origin duty, install month, termination"
              + (", fill base/value/sd" if pre else "") + ".",
              "- `county_class_monthly.parquet`: county and in-grid scopes; classes, all, "
              "all_nodomestic; value, sd, flag, method, hp_installed, hp_missing, "
              "missing_expected_kwh, n_fill.",
              "- `build_info.json`.", "",
              "Flags:",
              "- 0 observed. The pump-level method code separates observed, pre_install "
              "structural zero, observed_before_install_date, terminated_trailing_run, "
              "bill_in_second_month and uneven_pair. The long table's method is only "
              "'observed'.",
              "- 4: the December-2022 fill (pumps), or an aggregate holding it." if pre else
              "- 1: even-split alternative only.",
              "- 8: missing, never filled.",
              "- 9: suspect, an aggregate whose missing readings exceed 10 % of its installed "
              "kept HP" + ("." if pre else "; also monthly meters' 2026-07."),
              "", "Aggregates take the highest flag in the order 8 > 9 > 4 > 0. Missing "
              "post-install readings count as zero in aggregates, as in the model. "
              "`missing_expected_kwh` gives their expected kWh (HP x the same-month "
              "county-class mean kWh/HP of reporting pumps), as an indicator, not a fill; it "
              "is "
              f"{r10[name]['missing_expected_share_2012_2022']:.2%} of the 2012-2022 "
              "in-grid energy.", ""]
        v = r10[name]
        L += ["In-grid GWh/yr 2012-2022: " + ", ".join(
            f"{t} {v[t]['in_grid_GWh_per_yr_2012_2022']:.2f} (domestic "
            f"{v[t]['domestic_share_2012_2022']:.1%})" for t in lib.COUNTIES) + ".", ""]
        if pre:
            y3 = r3["v3"]
            L += ["**Yunlin against the calibrated model's forcing** (R3, reported, not "
                  "gating). v3 departs from the model by design:",
                  f"- 2012-01..2022-11: max monthly fan-total relative difference "
                  f"{y3['max_rel_month_2012_01_2022_11']:.1e} ({y3['worst_month_2012_01_2022_11']}), "
                  "from the kept-set flips;",
                  f"- 2022-12: {y3['delta_fan_total_2022_12_kwh'] / y3['model_fan_total_2022_12_kwh']:+.2%}"
                  " (v3 minus model), from the fill.",
                  "v1 reproduces the model's forcing (T2b), but it carries the leak.", ""]
    resid_txt = (
        "Their 2022-12 value is kept as read:" if name != "census_L1" else
        "Their 2022-12 reading enters the duty statistic as read:")
    resid_end = ("The full variant records 2022-12 as read, Dec-Jan halves included, so this "
                 "is not an exemption there." if name == "full" else
                 "The pre-origin and L1 manifests declare this as an exemption with its "
                 "sensitivity.")
    L += ["**Residual post-origin information.** Some meters are not labelled Dec-Jan-paired "
          "by the pre-origin rule, but their December-2022 and January-2023 readings are "
          "equal non-zero halves (found after the fact with the January-2023 reading). "
          + resid_txt,
          f"- Yunlin {pc['yunlin']['untreated_meters_kept_in_grid']:,} kept in-grid "
          f"meters, {pc['yunlin']['share_of_dec2022_in_grid']:.2%} of the in-grid "
          "December-2022 energy;",
          f"- Changhua {pc['changhua']['untreated_meters_kept_in_grid']:,} meters, "
          f"{pc['changhua']['share_of_dec2022_in_grid']:.2%};",
          f"- Chiayi {pc['chiayi']['untreated_meters_kept_in_grid']} meters.",
          "Leaving their December value out of the duty statistic flips no kept meter "
          "(`repair3_residual.py`). " + resid_end, ""]
    fc = facts["per_county"]
    bims = {}
    for t in lib.COUNTIES:
        pcc = binfo["counties"][t]["phase_code_counts"]
        bims[t] = (pcc.get("A", 0) + pcc.get("B", 0)) / sum(pcc.values())
    bim = (f"{bims['yunlin']:.1%} (Yunlin), {bims['changhua']:.1%} (Changhua) and "
           f"{bims['chiayi']:.1%} (Chiayi) of kept meters are billed bimonthly")
    key = "2021-12/2022-01"
    m1 = ", ".join(f"{fc[t]['m1_dec_jan_pair_equality'][key]['frac_equal']:.1%} ({t})"
                   for t in lib.COUNTIES)
    m1_other = min(v["frac_equal"] for t in lib.COUNTIES
                   for k, v in fc[t]["m1_dec_jan_pair_equality"].items() if k != key)
    nocoord = ", ".join(f"{t} {fc[t]['rows_without_coordinates']}" for t in lib.COUNTIES)
    kwh = name != "census_L1"      # bullets about kWh values do not apply to the L1 table
    lim = [(True, "- The census attribute table is the 2026 delivery and includes pumps "
                  "installed after the origin. This is a declared exemption of the pre-origin "
                  "and L1 products, with the R9 sensitivity."),
           (kwh, f"- {bim}, and TPC splits each bill evenly over its two months (flag 0, "
                 "method observed). Monthly values of bimonthly meters are therefore "
                 "even-split halves. No monthly-disaggregation test may score against them, "
                 "and this product allows neither scoring_target nor selection."),
           (kwh, f"- In the M1 pair 2021-12/2022-01 only {m1} of meter pairs are equal "
                 f"halves, against >= {m1_other:.1%} in every other year 2012-2024. The "
                 "2021-12 and 2022-01 values are TPC's uneven split. Pair totals are "
                 "unaffected, and the pump-level method labels them."),
           (True, "- Changhua's domestic class may carry household load. Use the "
                  "domestic-excluded variant as the sensitivity member."),
           (True, "- Pre-install coding is as in v1: the duty statistic sees the L0 readings "
                  "as delivered (Yunlin's pre-install zeros enter its mean, Changhua's and "
                  "Chiayi's NaN do not)."),
           (True, "- The TPC reading dates are undocumented. A reading taken in the first days "
                  "of January would add a few days of January to Nov-Dec pairs and to monthly "
                  "meters' December; this cannot be resolved from the data."),
           (True, f"- Census rows without coordinates ({nocoord}) have no cell and are left "
                  "out of the cell arrays, the model's convention."),
           (kwh, "- Coverage of TPC 農林漁牧 sales is about 0.4 and is not stationary "
                 "(descriptive, v1 T4).")]
    L += ["Known limits:"] + [txt for ok, txt in lim if ok]
    L += ["",
          f"Allowed uses: {', '.join(rf.ALLOWED[name])}."
          + (" Pre-origin uses take this variant, through its two declared exemptions."
             if pre else (" Recorded-forcing diagnostic arm only (U4)." if name == "full"
                          else " Pre-origin uses go through its two declared exemptions.")), "",
          f"Personal data: {rf.PERSONAL[name]}.", "",
          f"Pre-registration: {PREREG_ID}. Acceptance is in validation.json. WP report: "
          "results/dataprep/reports/C0.md (Repair round).", ""]
    return "\n".join(L)


def main() -> int:
    prods = {k: lib.REPO / v for k, v in lib.load_json(EV3 / "products_v3.json").items()}
    summ = lib.load_json(CHECKS / "summary.json")
    keys = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "T5")
    ev = {k: lib.load_json(CHECKS / f"{k}.json") for k in keys}
    res = lib.load_json(RESIDUAL)
    resume = lib.load_json(RESUME)
    facts = lib.load_json(FACTS)
    sha = ev["R1"]["prereg_sha256"]
    if any(t.get("prereg_sha256") != sha for t in ev.values()):
        raise SystemExit("check files carry different pre-registration hashes")
    if not (resume["checks_ok"] and resume["prereg_status"]["allowed"]):
        raise SystemExit("resume integrity check not clean")
    written = {}
    for name, d in prods.items():
        if (d / "MANIFEST.json").exists():
            raise SystemExit(f"{d} already has a manifest; never overwrite a version")
        g = summ["gates"][f"pump_electricity/{name}"]
        gate = {"passed": bool(g["passed"]),
                "criteria": f"pre-registered {PREREG_ID}: conjunction of "
                            + ", ".join(f"{k}={v}" for k, v in g["tests"].items()),
                "tests": g["tests"]}
        keep = {"census_L1": {"R1", "R2", "R4", "R5", "R6", "R9"},
                "preorigin": set(keys), "full": {"R1", "R4", "R5", "R6", "T5", "R10"}}[name]
        val = {"prereg": PREREG_ID, "prereg_sha256": sha, "gate": gate,
               "sd_label": summ.get("sd_label") if name == "preorigin" else None,
               "reported_only": summ["reported_only"], "evidence_dir": lib.rel(CHECKS),
               "evidence_sha256": {f"{k}.json": sha256_file(CHECKS / f"{k}.json") for k in ev},
               "tests": {k: {kk: vv for kk, vv in t.items()
                             if kk not in ("post_registration_looks", "files", "table")}
                         for k, t in ev.items() if k in keep},
               "supersedes": [f"pump_electricity/{name}/v2", f"pump_electricity/{name}/v1"],
               "superseded_note": "v2: registered gate failed (R4 check defect, T5 product "
                                  "defect); v1: the pre-origin variant carries January-2023 "
                                  "electricity (verifier finding, 2026-10-13)"}
        if name in ("preorigin", "census_L1"):
            val["exemption_sensitivity"] = {
                "residual_unpaired_dec2022": {"file": lib.rel(RESIDUAL),
                                              "sha256": sha256_file(RESIDUAL),
                                              "per_county": res["per_county"],
                                              "note": res["note"]},
                "census_attributes": {"file": lib.rel(CHECKS / "R9.json"),
                                      "sha256": sha256_file(CHECKS / "R9.json")}}
        val["card_facts"] = {"file": lib.rel(FACTS), "sha256": sha256_file(FACTS),
                             "note": "descriptive L0 facts quoted in the card (repair3_facts.py)"}
        val["resume_integrity_check"] = {"file": lib.rel(RESUME), "sha256": sha256_file(RESUME),
                                         "checks_ok": resume["checks_ok"],
                                         "prereg_allowed": resume["prereg_status"]["allowed"]}
        lib.save_json(val, d / "validation.json")
        binfo = lib.load_json(prods["preorigin"] / "build_info.json")
        (d / "PRODUCT_CARD.md").write_text(card(name, d, gate, ev, res, facts, binfo),
                                           encoding="utf-8")
        params = {"prereg": {"id": PREREG_ID, "sha256": sha},
                  "cleaning": "pumping.clean_census cap_duty 1.0, duty mean over 2012-01..2022-12 "
                              "of the L0 readings as delivered without the 2022-12 value of "
                              "Dec-Jan-paired meters; kept pumps' record re-shared with cap None",
                  "dec_jan_paired_rule": "identical-pair phase detected B on pairs inside "
                                         "2012-01..2022-11, or billing code M1 with detection "
                                         "in (mixed, insufficient, both-flat)",
                  "dec2022_fill": ("rho_bar(county, class) x (Oct+Nov 2022)/2, flag 4; base "
                                   "months coded as the product codes them (pre-install NaN or "
                                   "0 -> 0; positive readings before the install date kept); "
                                   "see fill_parameters.json" if name == "preorigin" else
                                   "none in this product (the pre-origin variant fills it)"
                                   if name == "census_L1" else
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
                  "supersedes": [f"pump_electricity/{name}/v2", f"pump_electricity/{name}/v1"],
                  "superseded_note": val["superseded_note"]}
        inputs = ([lib.rel(p) for p in rf.L0] + [lib.rel(lib.SOURCE_JSON), lib.rel(lib.GRID_NPZ)]
                  + [{"path": lib.rel(p), "role": "validation_reference: read only by the "
                      "acceptance checks (repair3_checks.py), never by the build"}
                     for p in rf.VALIDATION_REFS + V2_REFS]
                  + [{"path": lib.rel(DESIGN_LOOK), "role": "design_time_evidence: pre-origin "
                      "counts read before the Dec-Jan-paired rule was fixed (look "
                      "[C0-repair-design])"}])
        ex, ne = [], None
        if name in ("preorigin", "census_L1"):
            ex = [exemption_census(name, ev["R9"]), exemption_residual(name, res)]
            ne = "2022-12"
        m = manifest.build_manifest(
            d, product=f"pump_electricity/{name}", layer=rf.LAYER[name], inputs=inputs,
            code=[lib.rel(p) for p in CODE], parameters=params, fit_window_end="2026-07",
            validation_report="validation.json", gate=gate, allowed_uses=rf.ALLOWED[name],
            forbidden_uses=[u for u in rf.ALL_USES if u not in rf.ALLOWED[name]],
            personal_data=rf.PERSONAL[name], exemptions=ex, non_exempt_fit_window_end=ne)
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
    lib.save_json({"manifests": written, "privacy_rescan_hits": bad}, EV3 / "finalize.json")
    print(written, "privacy hits:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

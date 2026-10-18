"""Finalize a round-2 P2a-c product: copy the staging directory that passed its registered
construction checks into ``results/dataprep/yunlin_electricity/<variant>/v<N>/`` (a new version,
never an existing one), write ``validation.json``, ``PRODUCT_CARD.md`` and ``MANIFEST.json``.

Usage: python -I b4_finalize.py <variant> <tests_dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402

PREREG_ID = "P2abc.r2.c0_based.v1 (+ P2abc.r2.k1_repair.v1 for the construction checks)"
ZERO_WINDOW = {"preorigin": ("2012-01", "2022-12"), "full": ("2012-01", "2026-06")}
EVID = L.EVID
ZEROS = L.WORK / "zeros"
CODE = ["p2lib.py", "b1_build.py", "b2_zeros.py", "b3_assemble.py", "b4_finalize.py",
        "t1_meter_validation.py", "t2_head_check.py", "t3_july_backtest.py",
        "t4_zero_port.py", "t5_construction.py", "scoring.py", "prereg_v2.py", "prereg_k1r.py",
        "looks_p2.py"]
ALLOWED = {
    "preorigin": ["forcing_frozen", "model_design", "prior", "prior_width_input",
                  "design_time_evidence", "validator", "after_the_fact_analysis"],
    "full": ["forcing_recorded_arm", "statistical_layer_history", "validator",
             "after_the_fact_analysis"]}
FORBIDDEN = {
    "preorigin": ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                  "calibration_target", "initial_conditions", "idw_field", "selection",
                  "forcing_recorded_arm", "statistical_layer_history"],
    "full": ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
             "calibration_target", "initial_conditions", "idw_field", "selection",
             "forcing_frozen", "model_design", "prior", "prior_width_input",
             "design_time_evidence"]}
PERSONAL = ("well numbers (uid) in the per-pump arrays; no coordinates, names, addresses, "
            "postal codes or raw meter numbers; gitignored")


def inputs_for(variant: str) -> list[str]:
    d = L.C0_DIR[variant]
    ins = [L.rel(d / "MANIFEST.json"), L.rel(d / "cells_by_class.npz"),
           L.rel(d / "cells_total.npz"), L.rel(d / "county_class_monthly.parquet")]
    for c in L.COUNTIES:
        t = L.FILETAG[c]
        ins += [L.rel(d / f"pumps_{t}.npz"), L.rel(d / f"meters_{t}.parquet")]
    ins += [L.rel(L.C0_L1 / "MANIFEST.json"), L.rel(L.C0_L1 / "census_pumps.parquet")]
    ins += [L.rel(p) for p in sorted(L.WORK.glob(f"{variant}_*")) if p.is_file()]
    ins += [L.rel(ZEROS / f"zeros_{variant}.parquet"),
            L.rel(ZEROS / f"meter_zero_summary_{variant}.parquet")]
    tests = EVID / "tests"
    ins += [L.rel(tests / "t1_meter_validation_preorigin.json"),
            L.rel(tests / "t4_zero_port.json")]
    if variant == "full":
        ins += [L.rel(tests / "t1_meter_validation_full.json"),
                L.rel(tests / "t3_july_backtest.json")]
        ins += [L.rel(p) for p in sorted(L.WORK.glob("full_*")) if p.is_file()
                and L.rel(p) not in ins]
    return ins


def exemptions_preorigin() -> list[dict]:
    m, _ = manifest.load_manifest(L.C0_DIR["preorigin"])
    out = []
    for x in m["exemptions"]:
        kind = "census_attributes" if "census attribute" in x["input"] else \
            "residual_unpaired_dec2022"
        y = dict(x)
        y["ledger_number_id"] = f"P2abc.r2.exemption.{kind}.preorigin"
        y["inherited_from"] = f"pump_electricity/preorigin/v3 exemption {x['ledger_number_id']}"
        if kind == "residual_unpaired_dec2022":
            y["sensitivity"] = (x["sensitivity"] + " [P2 round 2: inherited unchanged; where "
                                "such a meter is detected phase A, the Chow-Lin arm re-splits "
                                "its Nov-Dec 2022 bill, so part of that December energy can "
                                "move into November 2022; the pair total is unchanged]")
        else:
            y["sensitivity"] = x["sensitivity"] + " [P2 round 2: inherited unchanged]"
        out.append(y)
    return out


def card(variant: str, d: Path, val: dict, bi: dict) -> str:
    t1 = val["P2r2.a1_a2"]
    lines = [f"# yunlin_electricity/{variant}/{d.name} (WP P2abc round 2)", "",
             f"Gate: **{'PASSED' if val['gate']['passed'] else 'FAILED'}** (pre-registered "
             f"{PREREG_ID}: {val['gate']['criteria']}).", "",
             "Despite its name (the plan's), this product covers every county with kept pumps "
             "in the twin's active cells: Yunlin, Changhua and the 316 kept in-grid Chiayi "
             "pumps. It builds on WP C0's `pump_electricity/" + variant + "/v3` and "
             "`census_L1/v3` (kept set, classes, cells, flags and method codes unchanged).", "",
             "Two arms, on C0's grid (2148 active cells) and time axis (index 0 = 2007-01; "
             + ("2007-01..2022-12" if variant == "preorigin" else "2007-01..2026-07") + "):",
             "- `recorded`: C0's values as read (TPC even split of each bimonthly bill)"
             + ("" if variant == "preorigin" else
                ", plus the P2b imputation of the open July-2026 phase-A month where shipped")
             + ";",
             "- `chowlin`: every eligible bimonthly irrigation bill re-split by a Chow-Lin "
             "disaggregation with the county's fan-wide profile (flag 3, method "
             "`chowlin_split`); pair totals are exact.", ""]
    lines += ["Shipping (registered rules):"]
    for c in L.COUNTIES:
        r = t1["counties"][c]
        lines.append(f"- {c}: Chow-Lin shipped {bi['ship_chowlin'][c]}; pre-origin meter "
                     f"skill min over n = 1..300 "
                     f"{r['min_skill_scored']:.3f} (n meters {r['n_validation_meters']}); "
                     f"sd label {bi['sd_label_chowlin'][c]}")
    if variant == "full":
        for c in L.COUNTIES:
            j = val["P2r2.b1"]["counties"][c]
            lines.append(f"- {c}: July-2026 imputation shipped {j['passed']} (backtest mean "
                         f"|fan-total error| {j['recorded_pair_mean']['mean_abs_total_err_pct']:.1f} %)")
    lines += ["", "Files:",
              "- `kwh_by_class_recorded.npz`, `kwh_by_class_chowlin.npz`: value/flag/sd (C, A, "
              "T); value_by_county/flag_by_county/sd_by_county (K, C, A, T); "
              "value/flag/sd/n_eff_total_all and _nodomestic (A, T); n_cl (flag-3 pumps), "
              "n_imp (imputed pumps), n_eff (effective number of Chow-Lin meters); C0's "
              "missing_expected_kwh arrays; grid metadata (mask_hash, grid_hash, nx, ny, dx, "
              "x0, y0).",
              "- `pumps_<county>.npz`: Chow-Lin arm per kept pump (uid, meter_idx, cell, "
              "purpose_class, hp, hp_share, value, flag, method, sd); same pump order as C0.",
              "- `county_class_monthly.parquet`: county x scope (in_grid, county) x class "
              "(+ all, all_nodomestic) x month x arm.",
              "- `meter_phase.parquet`: billing code, detected phase, Chow-Lin phase, rho, "
              "pairs re-split, profile member, chowlin_shipped.",
              "- `profiles.npz`: per-county fan-wide profiles and the phase-A/B population "
              "sums.",
              "- `incomplete_pair_mask.parquet` (the plan's july2025_mask): open bills at the "
              "record end, pairs with one missing month, bills posted to the second month.",
              "- `zero_class.parquet`, `meter_zero_summary.parquet`: P2c zero classes "
              "(window " + "..".join(ZERO_WINDOW[variant]) + ").",
              ]
    if variant == "full":
        lines.append("- `july2026_imputation.parquet`: imputed July-2026 phase-A values per "
                     "meter (both arms) with sd and the shipping flag.")
    lines += ["", "Flags: 0 observed (C0 methods); 3 Chow-Lin split; 4 C0's December-2022 fill "
              "(pre-origin) or the July-2026 imputation (full); 8 missing; 9 suspect. "
              "Aggregates: 8 > 9 > 4 > 3 > 0; a flag-8 aggregate is NaN (C0's rule). Missing "
              "readings count as zero in aggregates; C0's missing_expected_kwh indicates them.",
              "", "sd: flag 3 = q90(n_eff, calendar month)/1.645 x even-split energy of the "
              "Chow-Lin component, q90 from the validation groups of the county (Chiayi: "
              "Yunlin's); flag 4 = C0's fill sd or q90_rel/1.645 x the imputed energy; "
              "components in quadrature. Labels (registered P2r2.a2/b1): " +
              ", ".join(f"{c} Chow-Lin {bi['sd_label_chowlin'][c]}" for c in L.COUNTIES) +
              ". An 'uncalibrated' sd has not passed the plan 4.4 coverage rule (plan 4.7).",
              "", "Known limits:",
              "- The split is exact only in each bill's total; within-bill timing comes from "
              "the county-wide profile (flag 3). A single profile per county ignores local "
              "crop calendars.",
              "- The validation meters are monthly (M0/M9) meters, a different population "
              "(larger pumps) from the bimonthly ones.",
              "- Changhua's Chow-Lin sd under-covers in groups of 10 or more meters (registered "
              "coverage 0.86-0.88): it is labelled uncalibrated and too narrow.",
              "- Census frozen at mid-2016 installs; post-2016 pumps are P2d's synthetic "
              "layer, not here."]
    if variant == "preorigin":
        lines += ["- The December-2022 value of Dec-Jan-paired (phase-B) meters is C0's flag-4 "
                  "fill (the Dec 2022-Jan 2023 bill is open at the origin); it is not re-split.",
                  "- Inherits C0's two exemptions (2026 census attributes; residual unpaired "
                  "December-2022 halves), see the manifest."]
    else:
        lines += ["- From 2025-07 TPC often posts a whole bimonthly bill to its second month; "
                  "the recorded arm keeps that posting (C0's evensplit alternative re-splits "
                  "it evenly), the Chow-Lin arm re-splits it.",
                  "- 2026-07 is a partial month: phase-A bills are open (imputed or NaN), "
                  "monthly meters are flag 9."]
    lines += ["", f"Allowed uses: {', '.join(ALLOWED[variant])}.", "",
              f"Personal data: {PERSONAL}.", "",
              "Pre-registration: " + PREREG_ID + ". Acceptance: validation.json. WP report: "
              "results/dataprep/reports/P2abc.md."]
    return "\n".join(lines) + "\n"


def main(variant: str, tests_dir: str) -> None:
    td = Path(tests_dir)
    st = L.WORK / f"assembled_{variant}"
    k1_first = L.load_json(td / f"t5_construction_{variant}.json")
    k1 = L.load_json(td / f"t5_construction_{variant}_repair.json")
    c1 = L.load_json(td / "t4_zero_port.json")
    t1 = L.load_json(td / "t1_meter_validation_preorigin.json")
    t1v = L.load_json(td / f"t1_meter_validation_{variant}.json")
    bi = L.load_json(st / "build_info.json")
    gate_ok = bool(k1["passed"] and c1["passed"])
    d = manifest.new_version_dir(f"yunlin_electricity/{variant}")
    for p in sorted(st.iterdir()):
        shutil.copy2(p, d / p.name)
        assert sha256_file(p) == sha256_file(d / p.name)
    val = {"prereg": PREREG_ID, "variant": variant,
           "gate": {"passed": gate_ok,
                    "criteria": f"P2r2.k1r.construction_{variant} (P2abc.r2.k1_repair.v1, "
                                f"replacing P2r2.k1.construction_{variant}, which failed on "
                                f"the first assembly) and P2r2.c1.zero_port_yunlin",
                    "tests": {f"P2r2.k1r.construction_{variant}": k1["passed"],
                              "P2r2.c1.zero_port_yunlin": c1["passed"]}},
           "P2r2.k1r": {"violations": k1["violations"], "details": {
               k: v for k, v in k1["details"].items() if k != "h"}},
           "P2r2.k1_first_assembly_failed": {"passed": k1_first["passed"],
                                             "violations": k1_first["violations"],
                                             "item_f": k1_first["details"]["f"]},
           "P2r2.c1": c1["checks"],
           "P2r2.a1_a2": {"counties": {c: {k: t1["counties"][c].get(k) for k in (
               "n_validation_meters", "group_skill", "min_skill_scored", "passed",
               "placebo_skill")} | ({"sd_coverage": t1["counties"][c]["sd_coverage"]}
                                    if "sd_coverage" in t1["counties"][c] else {})
               for c in L.COUNTIES}},
           "sd_model_variant": {c: t1v["counties"][c].get("sd_coverage", {}).get("label")
                                for c in ("yunlin", "changhua")},
           "evidence": L.rel(td)}
    if variant == "full":
        t3 = L.load_json(td / "t3_july_backtest.json")
        val["P2r2.b1"] = {"counties": {c: {k: t3["counties"][c][k] for k in (
            "units", "recorded_pair_mean", "passed")} for c in L.COUNTIES}}
        val["P2r2.a1_full_variant_secondary"] = {
            c: {k: t1v["counties"][c].get(k) for k in ("n_validation_meters", "group_skill",
                                                        "min_skill_scored")}
            for c in L.COUNTIES}
    L.save_json(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card(variant, d, val, bi), encoding="utf-8")
    pre = variant == "preorigin"
    params = {"prereg": {"id": PREREG_ID}, "variant": L.VARIANTS[variant] | {"open": {
        k: L.ym(v) for k, v in L.VARIANTS[variant]["open"].items()}, "win_hi": {
        k: L.ym(v - 1) for k, v in L.VARIANTS[variant]["win_hi"].items()}, "cl0": L.ym(
        L.VARIANTS[variant]["cl0"])},
              "profile": {"lambda": L.LAMBDA, "share_window": L.SHARE_WIN,
                          "per_county": True, "chiayi_uses": "yunlin"},
              "chowlin": {"rhos": list(L.RHOS), "X": "[1, p / mean p]",
                          "eligible_methods": [L.METHODS[int(x)] for x in L.RESPLIT_OK]},
              "ship_chowlin": bi["ship_chowlin"], "sd_label_chowlin": bi["sd_label_chowlin"],
              "methods": {str(k): v for k, v in L.METHODS.items()},
              "zero_window": list(ZERO_WINDOW[variant])}
    if not pre:
        params["july2026_imputation"] = {c: bi["counties"][c]["q_imp"] for c in L.COUNTIES}
    m = manifest.build_manifest(
        d, product=f"yunlin_electricity/{variant}", layer="L2",
        inputs=inputs_for(variant), code=[L.rel(L.CODE_DIR / f) for f in CODE],
        parameters=params, fit_window_end="2026-07",
        non_exempt_fit_window_end="2022-12" if pre else None,
        exemptions=exemptions_preorigin() if pre else None,
        validation_report="validation.json", gate=val["gate"],
        allowed_uses=ALLOWED[variant], forbidden_uses=FORBIDDEN[variant],
        personal_data=PERSONAL)
    manifest.write_manifest(d, m)
    print("written", d, "gate", gate_ok)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

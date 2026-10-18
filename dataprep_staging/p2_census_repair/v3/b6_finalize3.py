"""Finalize a repair-round P2a-c product: copy the staging directory whose registered checks
passed into ``results/dataprep/yunlin_electricity/<variant>/v<N>/`` (a new version, never an
existing one; expected v2), write ``validation.json``, ``PRODUCT_CARD.md`` and
``MANIFEST.json``.

Usage: python -I b6_finalize3.py <variant>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import b4_finalize as B4  # noqa: E402  (v2, read-only: ALLOWED, FORBIDDEN, PERSONAL, ...)
import b5_assemble3 as B5  # noqa: E402
import p3lib as P3  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from p3lib import L  # noqa: E402

PREREG_ID = ("P2abc.r3.repair.v2 (repair round, amending P2abc.r3.repair.v1 before any of its "
             "tests was scored; + P2abc.r3.k1_repair.v1 for the full product's construction "
             "check; the round-2 registrations P2abc.r2.c0_based.v1 "
             "and P2abc.r2.k1_repair.v1 stand for the unchanged Chow-Lin, sd and July "
             "decisions)")
TESTS2 = L.EVID / "tests"
TESTS3 = P3.EVID3 / "tests"
ZEROS3 = P3.WORK3 / "zeros"
CODE3 = ["p3lib.py", "zeros3.py", "b5_assemble3.py", "b6_finalize3.py", "t6_zero_shipped.py",
         "t7_construction3.py", "t8_july_member.py", "prereg_r3.py", "prereg_r3b.py",
         "prereg_k1r3.py", "looks3.py"]
CODE2 = ["p2lib.py", "b1_build.py", "b2_zeros.py", "b3_assemble.py", "b4_finalize.py",
         "t1_meter_validation.py", "t3_july_backtest.py", "t5_construction.py", "scoring.py",
         "looks_p2.py"]
ALLOWED = B4.ALLOWED
FORBIDDEN = B4.FORBIDDEN
PERSONAL = B4.PERSONAL
ZERO_WINDOW = B4.ZERO_WINDOW


def inputs_for(variant: str) -> list[str]:
    d = L.C0_DIR[variant]
    ins = [L.rel(d / "MANIFEST.json"), L.rel(d / "cells_by_class.npz"),
           L.rel(d / "cells_total.npz"), L.rel(d / "county_class_monthly.parquet")]
    for c in L.COUNTIES:
        t = L.FILETAG[c]
        ins += [L.rel(d / f"pumps_{t}.npz"), L.rel(d / f"meters_{t}.parquet")]
    ins += [L.rel(L.C0_L1 / "MANIFEST.json"), L.rel(L.C0_L1 / "census_pumps.parquet")]
    ins += [L.rel(p) for p in sorted(L.WORK.glob(f"{variant}_*")) if p.is_file()]
    ins += [L.rel(ZEROS3 / f"zeros_{variant}.parquet"),
            L.rel(ZEROS3 / f"meter_zero_summary_{variant}.parquet")]
    ins += [L.rel(TESTS2 / "t1_meter_validation_preorigin.json"),
            L.rel(TESTS3 / "t6_zero_shipped.json")]
    if variant == "full":
        ins += [L.rel(TESTS2 / "t1_meter_validation_full.json"),
                L.rel(TESTS2 / "t3_july_backtest.json")]
        ins += [L.rel(p) for p in sorted(L.WORK.glob("full_*")) if p.is_file()
                and L.rel(p) not in ins]
    return ins


def extra_exemptions_preorigin() -> list[dict]:
    """The repair round's own exemption of the pre-origin product: the registered zero check
    (gate input) reads the Yunlin record to 2025-07."""
    p = L.rel(TESTS3 / "t6_zero_shipped.json")
    return [{
        "input": "result of the registered zero check P2r3b.c1.zero_shipped_yunlin "
                 "(reports/P2abc_evidence/repair3/tests/t6_zero_shipped.json), computed on "
                 "the C0 full v3 Yunlin record 2007-01..2025-07 in the billing pilot's view",
        "input_paths": [p],
        "data_end": "2025-07",
        "reason": "the plan's P2c acceptance is reproduction of the pilot's counts, which the "
                  "pilot computed on 2012-01..2025-06; the check can only be made on that window",
        "ledger_number_id": "P2abc.r3.exemption.zero_check_gate.preorigin",
        "sensitivity": "no value, flag, sd, zero class or parameter of the pre-origin product "
                       "depends on it: the zero tables are written by zeros3.classify on the "
                       "pre-origin record (window 2012-01..2022-12, record to 2022-12) before "
                       "the check runs, and construction check (i) re-runs the classifier and "
                       "reproduces them exactly; the check decides only whether the product's "
                       "gate passes (code identity with the pilot)",
        "uses": ["forcing_frozen", "model_design", "prior", "prior_width_input",
                 "design_time_evidence"]}]


def exemptions_preorigin() -> list[dict]:
    m, _ = manifest.load_manifest(L.C0_DIR["preorigin"])
    out = []
    for x in m["exemptions"]:
        kind = "census_attributes" if "census attribute" in x["input"] else \
            "residual_unpaired_dec2022"
        y = dict(x)
        y["ledger_number_id"] = f"P2abc.r3.exemption.{kind}.preorigin"
        y["inherited_from"] = f"pump_electricity/preorigin/v3 exemption {x['ledger_number_id']}"
        y["sensitivity"] = x["sensitivity"] + " [P2 repair round: inherited unchanged]"
        out.append(y)
    return out + extra_exemptions_preorigin()


def card(variant: str, d: Path, val: dict, bi: dict) -> str:
    pre = variant == "preorigin"
    lines = [f"# yunlin_electricity/{variant}/{d.name} (WP P2abc repair round)", "",
             f"Gate: **{'PASSED' if val['gate']['passed'] else 'FAILED'}** "
             f"({val['gate']['criteria']}).", "",
             "**Coverage: despite the plan's name, this product covers every county with kept "
             "pumps in the twin's active cells: Yunlin, Changhua and the 316 kept in-grid "
             "Chiayi pumps.** It builds on WP C0's `pump_electricity/" + variant + "/v3` and "
             "`census_L1/v3`. It supersedes `" + variant + "/v1` (round 2), whose zero table "
             "came from a classifier the registered check did not score" +
             ("" if pre else " and whose 2025 handling missed the plan's M0 July-2025 flag")
             + ".", "",
             "Arms (C0's grid of 2148 active cells; index 0 = 2007-01; " +
             ("2007-01..2022-12" if pre else "2007-01..2026-07") + "):",
             "- `recorded`: C0's values as read (TPC even split of each bimonthly bill)" +
             ("" if pre else ", plus the P2b imputation of the open July-2026 phase-A month "
              "where shipped") + ";",
             "- `chowlin`: eligible bimonthly irrigation bills re-split by a Chow-Lin "
             "disaggregation with the county's fan-wide profile (flag 3); pair totals exact;",
             "- `recorded_syncfill`, `chowlin_syncfill` (sensitivity members, plan P2c): the "
             "arm with every synchronized-zero meter-month replaced by the meter's own mean of "
             "the same calendar month over its three most recent eligible earlier years "
             "(flag 4, method sync_zero_fill; sd uncalibrated). The primary arms keep the "
             "recorded zeros.", ""]
    lines += ["Shipping (registered round-2 rules, unchanged):"]
    for c in L.COUNTIES:
        lines.append(f"- {c}: Chow-Lin shipped {bi['ship_chowlin'][c]}; sd label "
                     f"{bi['sd_label_chowlin'][c]}")
    if not pre:
        for c in L.COUNTIES:
            j = val["P2r2.b1"]["counties"][c]
            lines.append(f"- {c}: July-2026 imputation shipped {j['passed']} (backtest mean "
                         f"|fan-total error| "
                         f"{j['recorded_pair_mean']['mean_abs_total_err_pct']:.1f} %)")
    lines += ["", "Files:",
              "- `kwh_by_class_<arm>.npz` (4 arms): value/flag/sd (C, A, T); value_by_county/"
              "flag_by_county/sd_by_county (K, C, A, T); value/flag/sd/n_eff_total_all and "
              "_nodomestic (A, T); n_cl, n_imp, n_sync (flag-3, imputed, filled pumps), n_eff; "
              "C0's missing_expected_kwh arrays; grid metadata.",
              "- `pumps_<county>.npz`: Chow-Lin arm per kept pump (uid, meter_idx, cell, "
              "purpose_class, hp, hp_share, value, flag, method, sd); C0's pump order.",
              "- `county_class_monthly.parquet`: county x scope x class x month x arm (4 arms).",
              "- `meter_phase.parquet`, `profiles.npz`: as v1.",
              "- `incomplete_pair_mask.parquet`: open bills at the record end, one-month-missing "
              "pairs, bills posted to the second month, synchronized-zero months (reason "
              "sync_zero_month)" + ("" if pre else ", M0 July-2025 suspects") + ".",
              "- `zero_class.parquet`: one row per zero meter-month of the window ("
              + "..".join(ZERO_WINDOW[variant]) + "): zkind (the billing pilot's run kind: "
              "leading, interior, trailing, no_use_in_window), run_len, zclass (not_yet_used, "
              "idle, long_idle, terminated, trailing_short, never_used_in_window), structural "
              "(C0 pre_install), sync_suspect, flag (0 or 9), possible_replacement.",
              "- `meter_zero_summary.parquet`: first/last use, terminated (a trailing zero run "
              ">= 12 months), last-12-month energy before termination, C0's flag.",
              "- `sync_fill.parquet`: the members' fill per meter-month (meter kWh; pump = "
              "hp_share x meter) with sd."]
    if not pre:
        lines.append("- `july2026_imputation.parquet`: July-2026 phase-A imputations per meter "
                     "(arms and members) with sd and the shipping flag.")
    lines += ["", "Flags: 0 observed; 3 Chow-Lin split; 4 C0's December-2022 fill (pre-origin), "
              "the July-2026 imputation (full) or a synchronized-zero fill (members); 8 "
              "missing; 9 suspect" + ("" if pre else " (C0's partial 2026-07 cycle; M0 meters "
                                      "in 2025-07, plan P2b: an aggregate is 9 when such pumps "
                                      "hold > 10 % of its installed HP)") + ". Aggregates: "
              "8 > 9 > 4 > 3 > 0; a flag-8 aggregate is NaN. Missing readings count as zero in "
              "aggregates (C0's missing_expected_kwh indicates them).",
              "", "sd labels: Yunlin Chow-Lin calibrated; Changhua uncalibrated (too narrow at "
              "groups >= 10 meters); Chiayi uncalibrated (not checked on Chiayi meters; uses "
              "Yunlin's model); " + ("C0's December-2022 fill sd uncalibrated (C0 verifier); "
                                     if pre else "July-2026 imputation uncalibrated; ") +
              "synchronized-zero fill sd uncalibrated (in-sample q90).",
              "", "Known limits:",
              "- Within-bill timing of the Chow-Lin arm comes from one county-wide profile; "
              "validation meters (monthly M0/M9) are a different population.",
              "- The synchronized-zero months' cause is unverified (plan P2c); the members bound "
              "their effect, the primary arms keep the recorded zeros.",
              "- Census frozen at mid-2016 installs (post-2016 pumps: P2d)."]
    if pre:
        lines += ["- December 2022 of Dec-Jan-paired meters is C0's flag-4 fill; three "
                  "exemptions (two inherited from C0, one for the zero check), see the "
                  "manifest."]
    else:
        lines += ["- From 2025-07 TPC often posts a whole bimonthly bill to its second month; "
                  "the recorded arm keeps it, the Chow-Lin arm re-splits it.",
                  "- Yunlin's phase-A July-August 2025 bills show the synchronized-zero "
                  "pattern of 2019 and 2021 (zero share 0.164 of active meters); they are "
                  "recorded zeros in the primary arms and filled in the members."]
    lines += ["", f"Allowed uses: {', '.join(ALLOWED[variant])}.", "",
              f"Personal data: {PERSONAL}.", "",
              "Pre-registration: " + PREREG_ID + ". Acceptance: validation.json. WP report: "
              "results/dataprep/reports/P2abc.md (section 'Repair round')."]
    return "\n".join(lines) + "\n"


def main(variant: str) -> None:
    st = P3.WORK3 / f"assembled_{variant}"
    k1_first = None
    k1_id = f"P2r3b.k1.construction_{variant}"
    if variant == "full":           # failed first assembly; repair P2abc.r3.k1_repair.v1
        k1_first = L.load_json(TESTS3 / "t7_construction3_full.json")
        k1 = L.load_json(TESTS3 / "t7_construction3_full_repair.json")
        k1_id = "P2r3c.k1.construction_full"
    else:
        k1 = L.load_json(TESTS3 / f"t7_construction3_{variant}.json")
    c1 = L.load_json(TESTS3 / "t6_zero_shipped.json")
    t1 = L.load_json(TESTS2 / "t1_meter_validation_preorigin.json")
    t1v = L.load_json(TESTS2 / f"t1_meter_validation_{variant}.json")
    bi = L.load_json(st / "build_info.json")
    gate_ok = bool(k1["passed"] and c1["passed"])
    d = manifest.new_version_dir(f"yunlin_electricity/{variant}")
    for p in sorted(st.iterdir()):
        shutil.copy2(p, d / p.name)
        assert sha256_file(p) == sha256_file(d / p.name)
    val = {"prereg": PREREG_ID, "variant": variant,
           "gate": {"passed": gate_ok,
                    "criteria": f"{k1_id} and P2r3b.c1.zero_shipped_yunlin (P2abc.r3.repair.v2"
                                + (", P2abc.r3.k1_repair.v1 for the construction check, "
                                   "replacing P2r3b.k1.construction_full, which failed on the "
                                   "first assembly)" if variant == "full" else ")"),
                    "tests": {k1_id: k1["passed"],
                              "P2r3b.c1.zero_shipped_yunlin": c1["passed"]}},
           "construction": {"test_id": k1_id, "violations": k1["violations"], "details": {
               k: v for k, v in k1["details"].items() if k != "h"}},
           "P2r3b.c1": c1["checks"],
           "P2r3b.c1_reported": c1["reported"],
           "P2r2.a1_a2": {"counties": {c: {k: t1["counties"][c].get(k) for k in (
               "n_validation_meters", "group_skill", "min_skill_scored", "passed",
               "placebo_skill")} | ({"sd_coverage": t1["counties"][c]["sd_coverage"]}
                                    if "sd_coverage" in t1["counties"][c] else {})
               for c in L.COUNTIES}},
           "sd_model_variant": {c: t1v["counties"][c].get("sd_coverage", {}).get("label")
                                for c in ("yunlin", "changhua")},
           "sd_label_chowlin": bi["sd_label_chowlin"],
           "syncfill": {c: {a: bi["counties"][c]["arms"][a].get("fill") for a in
                            ("recorded_syncfill", "chowlin_syncfill")} for c in L.COUNTIES},
           "evidence": [L.rel(TESTS2), L.rel(TESTS3)]}
    if k1_first is not None:
        val["P2r3b.k1_first_assembly_failed"] = {"passed": k1_first["passed"],
                                                 "violations": k1_first["violations"],
                                                 "item_c": k1_first["details"]["c"]}
    if variant == "full":
        t3 = L.load_json(TESTS2 / "t3_july_backtest.json")
        val["P2r2.b1"] = {"counties": {c: {k: t3["counties"][c][k] for k in (
            "units", "recorded_pair_mean", "passed")} for c in L.COUNTIES}}
        p8 = TESTS3 / "t8_july_member.json"
        if p8.exists():
            t8 = L.load_json(p8)
            val["P2r3b.b2_reported"] = {c: {k: t8["counties"][c][k] for k in (
                "member_2013_2025", "member_2013_2024", "recorded_2013_2024")}
                for c in L.COUNTIES}
        val["m0_july2025_suspect"] = {c: {k: bi["counties"][c][k] for k in (
            "m0_july2025_suspect_meters", "m0_july2025_suspect_pumps")} for c in L.COUNTIES}
    L.save_json(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card(variant, d, val, bi), encoding="utf-8")
    pre = variant == "preorigin"
    params = {"prereg": {"id": PREREG_ID},
              "variant": L.VARIANTS[variant] | {
                  "open": {k: L.ym(v) for k, v in L.VARIANTS[variant]["open"].items()},
                  "win_hi": {k: L.ym(v - 1) for k, v in L.VARIANTS[variant]["win_hi"].items()},
                  "cl0": L.ym(L.VARIANTS[variant]["cl0"])},
              "arms": list(B5.ARMS),
              "profile": {"lambda": L.LAMBDA, "share_window": L.SHARE_WIN,
                          "per_county": True, "chiayi_uses": "yunlin"},
              "chowlin": {"rhos": list(L.RHOS), "X": "[1, p / mean p]",
                          "eligible_methods": [L.METHODS[int(x)] for x in L.RESPLIT_OK]},
              "ship_chowlin": bi["ship_chowlin"], "sd_label_chowlin": bi["sd_label_chowlin"],
              "methods": bi["methods"], "zero_window": list(ZERO_WINDOW[variant]),
              "zero_classifier": "zeros3.classify (billing pilot s05(e) run kinds; structural "
                                 "column = C0 pre_install; pair-consistent synchronized-zero "
                                 "marks)",
              "syncfill": {"rule": "own mean of the same calendar month over the 3 most recent "
                                   "eligible earlier years (<= 6 years back)",
                           "q90_by_county_arm": {c: {a: bi["counties"][c]["arms"][a]["fill"][
                               "q90_by_moy"] for a in ("recorded_syncfill", "chowlin_syncfill")}
                               for c in L.COUNTIES}}}
    if not pre:
        params["july2026_imputation"] = {c: bi["counties"][c]["q_imp"] for c in L.COUNTIES}
        params["m0_july2025"] = {"rule": "code M0, detected phase monthly, flag-0 reading in "
                                         "2025-07 -> flag 9, method m0_july2025_suspect (value "
                                         "kept); aggregate flag 9 when > 10 % of installed HP",
                                 "source": "plan P2b"}
    code = [L.rel(P3.V3 / f) for f in CODE3] + [L.rel(L.CODE_DIR / f) for f in CODE2]
    m = manifest.build_manifest(
        d, product=f"yunlin_electricity/{variant}", layer="L2",
        inputs=inputs_for(variant), code=code, parameters=params, fit_window_end="2026-07",
        non_exempt_fit_window_end="2022-12" if pre else None,
        exemptions=exemptions_preorigin() if pre else None,
        validation_report="validation.json", gate=val["gate"],
        allowed_uses=ALLOWED[variant], forbidden_uses=FORBIDDEN[variant],
        personal_data=PERSONAL)
    manifest.write_manifest(d, m)
    print("written", d, "gate", gate_ok)


if __name__ == "__main__":
    main(sys.argv[1])

"""WP C0 repair round: update the WP report (results/dataprep/reports/C0.md and C0.json) with a
'Repair round' section and append the ledger rows (plan 4.1, U6) of every number that section
quotes. Reads only evidence files already written (no product is opened for computation).

* C0.md: new header (status, current products v3), round-1 corrections marked in place (the
  billing-code agreement figures, the sd sentence), section 7 'Repair round' appended (replaced
  if the script is rerun).
* C0.json: the WP as it now stands (v3 acceptance, products, deviations, issues, key numbers);
  the round-1 content is kept under ``round_1``.
* ledger/rows/C0.csv: rows for each number id quoted (exact duplicates are skipped), then a
  conflict check over every fragment and an unledgered check on the ids quoted.

Usage: python -I repair3_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair_ledger as rlg  # noqa: E402  (rows_for, l0, L0_ALL, v2_rows; not its main)
from common import ledger  # noqa: E402

EV = lib.EVIDENCE
EV2, EV3 = EV / "repair", EV / "repair3"
CK2, CK3 = EV2 / "checks", EV3 / "checks"
REPORTS = lib.DATAPREP / "reports"
MD, JS = REPORTS / "C0.md", REPORTS / "C0.json"
PE = "results/dataprep/pump_electricity"
PRE, FULL, L1 = f"{PE}/preorigin/v3", f"{PE}/full/v3", f"{PE}/census_L1/v3"
PRE2, FULL2, L12 = f"{PE}/preorigin/v2", f"{PE}/full/v2", f"{PE}/census_L1/v2"
PRE1, FULL1, L11 = f"{PE}/preorigin/v1", f"{PE}/full/v1", f"{PE}/census_L1/v1"
PKL = "results/twin_runs/stage3_datum_gate_cwa/inputs.local.pkl"
GRID = "results/twin_forward/datum_gate_innov.npz"
C = lib.COUNTIES
NAME = {"yunlin": "Yunlin", "changhua": "Changhua", "chiayi": "Chiayi"}
QUOTED: list[str] = []
ROWS: list[dict] = []


def q(number_id: str, version: str, roles: dict, note: str) -> str:
    """Register a quoted number with its ledger rows; returns the id (for the JSON)."""
    if number_id not in QUOTED:
        QUOTED.append(number_id)
        ROWS.extend(rlg.rows_for(number_id, version, roles, note))
    return number_id


def j(p: Path) -> dict:
    return lib.load_json(p)


def pct(x: float, n: int = 1) -> str:
    return f"{100 * x:.{n}f} %"


def load() -> dict:
    e = {k: j(CK3 / f"{k}.json") for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
                                           "R9", "R10", "T5", "summary")}
    e["v2"] = {k: j(CK2 / f"{k}.json") for k in ("R1", "R2", "R4", "T5", "summary")}
    e["design"] = j(EV2 / "design_look.json")
    e["resid"] = j(EV3 / "residual_sensitivity.json")
    e["facts"] = j(EV3 / "report_facts.json")
    e["resume"] = j(EV3 / "resume_check.json")
    e["probe"] = j(EV3 / "guard_probe.json")
    e["finalize"] = j(EV3 / "finalize.json")
    e["fill"] = j(lib.REPO / PRE / "fill_parameters.json")
    e["prereg_sha"] = e["R1"]["prereg_sha256"]
    looks = []
    with open(lib.DATAPREP / "preregistration" / "looks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("wp") == lib.WP and (str(r.get("note", "")).startswith("[C0-repair")
                                          or str(r.get("test_id") or "").startswith(
                                              ("C0.v2.", "C0.v3."))):
                looks.append(r)
    e["looks"] = looks
    return e


# ----------------------------------------------------------------------------- markdown
def section7(e: dict) -> str:
    r1, r2, r3, r7, r8, r9, r10 = (e[k] for k in ("R1", "R2", "R3", "R7", "R8", "R9", "R10"))
    res, fa = e["resid"]["per_county"], e["facts"]["per_county"]
    bi = r10["preorigin"]["counties_build"]
    v2 = e["v2"]
    out = ["## 7. Repair round (2026-10-13, resumed 2026-10-14)", ""]

    # 7.1 the finding
    q("C0.v3.R8.fill_vs_actual.all", PRE, {"input": [PRE], "validator": rlg.L0_ALL + [PRE1]},
      "v1 in-grid Dec-2022 kWh and the paired meters' share of it (R8): " + ", ".join(
          f"{t} {r8['per_county'][t]['dec2022_in_grid_v1_kWh']:.0f} kWh, paired share "
          f"{r8['per_county'][t]['v1_paired_share_of_dec2022']:.4f}" for t in C))
    v1dec = sum(r8["per_county"][t]["dec2022_in_grid_v1_kWh"] for t in C) / 1e6
    out += ["### 7.1 The verifier's finding", "",
            "The verifier found one critical issue: a U3/U4 breach in `preorigin/v1`. A "
            "phase-B (M1) meter's December value is half of a December-January bill, so the "
            "2022-12 value of every such meter is half of a bill read in 2023. That January-2023 "
            "information sat in the frozen forcing and in the duty statistic that selects the "
            "kept set. The manifest declared no exemption.",
            "",
            "Measured after the fact (R8), the meters the repair labels Dec-Jan-paired carried "
            + ", ".join(f"{pct(r8['per_county'][t]['v1_paired_share_of_dec2022'])} ({NAME[t]})"
                        for t in C)
            + f" of v1's in-grid December-2022 energy ({v1dec:.2f} GWh in all). Round 1 "
            "disclosed this as deviation 3; it is now repaired. The verifier also raised 11 "
            "minor points (section 7.5).", ""]

    # 7.2 what changed
    for t in C:
        b = bi[t]
        q(f"C0.v3.R10.build_counts.{t}", PRE, {"input": rlg.l0(t) + [PRE]},
          f"Dec-Jan-paired kept meters {b['meters_dec_jan_paired']}, filled "
          f"{b['meters_filled_dec2022']}, without base {b['meters_paired_without_base']}")
        q(f"C0.v3.R1.flips.{t}", PRE, {"input": rlg.l0(t) + [PRE, L1], "validator": [PRE1, L11]},
          f"kept-meter flips v3 vs v1: +{r1['per_county'][t]['flips_in_meters']} / "
          f"-{r1['per_county'][t]['flips_out_meters']}")
    fp = e["fill"]["counties"]
    q("C0.v3.fill_parameters", PRE, {"input": rlg.L0_ALL + [PRE]},
      "rho_bar irrigation: " + ", ".join(
          f"{t} {fp[t]['classes']['irrigation']['rho_bar']:.4f} (s "
          f"{fp[t]['classes']['irrigation']['s']:.4f}), sigma {fp[t]['sigma']:.4f}" for t in C))
    q("C0.v3.R10.dec2022", PRE, {"input": rlg.L0_ALL + [PRE]},
      f"v3 Dec-2022 in-grid {r10['preorigin']['dec2022_in_grid_kWh']:.0f} kWh, filled cells "
      f"{r10['preorigin']['dec2022_filled_cells']}, sd upper "
      f"{r10['preorigin']['dec2022_sd_fan_total_upper_kWh']:.0f} kWh")
    q("C0.v3.R10.missing_expected", PRE, {"input": rlg.L0_ALL + [PRE]},
      f"missing_expected share 2012-2022 {r10['preorigin']['missing_expected_share_2012_2022']:.5f}")
    out += ["### 7.2 What changed (v3 products)", "",
            "v3 removes the post-origin information from the pre-origin variant everywhere "
            "and leaves the cleaning rules unchanged. The v2 design (`repair_lib.py`) was "
            "fixed before any comparison with a reference was read. v3 (`repair3_lib.py`) "
            "changed one rule, the fill base, after v2's registered T5 failure (section "
            "7.3).", "",
            "- **Dec-Jan-paired meters.** A meter is Dec-Jan-paired if its identical-pair phase, "
            "detected only on pairs lying wholly inside 2012-01..2022-11, is B, or if its code "
            "is M1 and the detection is inconclusive. The rule was chosen after a design look "
            "at pre-origin counts only ([C0-repair-design]). Kept meters labelled paired: "
            + ", ".join(f"{NAME[t]} {bi[t]['meters_dec_jan_paired']:,}" for t in C) + ".",
            "- **Duty statistic and kept set.** The mean 2012-2022 draw now leaves out the "
            "2022-12 value of paired meters. Both variants and the L1 table share this "
            "pre-origin kept set. Kept meters against v1: "
            + ", ".join(f"{NAME[t]} +{r1['per_county'][t]['flips_in_meters']}/"
                        f"-{r1['per_county'][t]['flips_out_meters']}" for t in C) + ".",
            "- **December-2022 fill**, pre-origin variant only: flag 4, method "
            "`fill_dec2022_dec_jan_pair`.",
            "  - Value: rho_bar(county, class) x (Oct-2022 + Nov-2022)/2. rho_bar is the mean "
            "over 2012-2021 of sum(Dec_y + Jan_y+1) / sum(Oct_y + Nov_y) over the county's "
            "paired kept meters of that class, so every reading it uses lies in "
            "2012-10..2022-01.",
            "  - Irrigation rho_bar (inter-annual sd s): "
            + ", ".join(f"{NAME[t]} {fp[t]['classes']['irrigation']['rho_bar']:.3f} "
                        f"({fp[t]['classes']['irrigation']['s']:.3f})" for t in C) + ".",
            "  - Filled meters: " + ", ".join(f"{NAME[t]} {bi[t]['meters_filled_dec2022']:,}"
                                              for t in C)
            + ". Without an Oct-Nov 2022 base the value stays missing (flag 8): "
            + ", ".join(f"{bi[t]['meters_paired_without_base']}" for t in C) + " meters.",
            "  - v3 codes the fill base as the product codes those months: a pre-install NaN "
            "or 0 is a structural zero, and a positive reading before the census install date "
            "is kept. This fixed v2's T5 product defect.",
            "- **sd.** The fill carries an sd made of a fully correlated ratio factor and a "
            "meter term (sigma "
            + ", ".join(f"{fp[t]['sigma']:.2f}" for t in C) + "). Its label is set by R7. "
            "Observed values keep sd NaN, because the TPC reading error is not modelled.",
            f"- **`missing_expected_kwh`** is new. It gives the expected kWh of post-install "
            f"missing readings, which still count as zero, and is "
            f"{pct(r10['preorigin']['missing_expected_share_2012_2022'], 2)} of 2012-2022 "
            "in-grid energy. It is an indicator, not a fill.",
            f"- **December 2022 in grid:** v1 {v1dec:.2f} GWh, v3 "
            f"{r10['preorigin']['dec2022_in_grid_kWh'] / 1e6:.2f} GWh. The sd of the fan "
            f"total is <= {r10['preorigin']['dec2022_sd_fan_total_upper_kWh'] / 1e6:.2f} GWh.",
            "- **Allowed uses.** `preorigin/v3` no longer allows `scoring_target` or "
            "`selection`, because flag 0 also covers TPC even-split halves and pre-install "
            "zeros (minor finding 2). The guard probe confirms both are refused.",
            "- **Exemptions.** `preorigin/v3` and `census_L1/v3` declare two exemptions "
            "(plan 2.3): fit_window_end 2026-07, non_exempt_fit_window_end 2022-12. Each has "
            "a ledger id and a sensitivity run (section 7.4).", ""]

    # 7.3 registered tests
    out += ["### 7.3 Registered tests", "",
            "**C0.census_product.v2** (created 2026-10-13T08:09:54Z, sha256 `33067d13…`) "
            "**failed its gate**:",
            f"- R4 failed with {v2['R4']['preorigin']['violations']:,} violations, all in the sd "
            "recomputation. This was a check defect: the check used the pump's class where "
            "the product uses the meter's class. With the meter class it matches to 9e-13 "
            "([C0-repair-diag-1]).",
            f"- T5 failed with {v2['T5']['violations']} class x cell entry at 2022-12 "
            f"({v2['T5']['items']['cells_dec2022_without_fill']['max_abs']} kWh). This was a "
            "product defect: a Changhua paired meter read before its census install date got "
            "no fill ([C0-repair-diag-2]).",
            "- R1, R2, R5 and R6 passed.", "",
            "The v2 products are published with gate False and must not be used. v3 changes "
            "the fill base rule (product defect) and corrects the R4 and T5 checks (check "
            "defects). These decisions were logged ([C0-repair-v3-decision]) before any v2 R3, "
            "R7, R8 or R10 result was read.", "",
            f"**C0.census_product.v3** (created 2026-10-13T08:29:16Z, sha256 "
            f"`{e['prereg_sha'][:8]}…`, harness.prereg3): every test was scored with the same "
            "checks-code sha256 `a89034d0…`.", "",
            "| test | criterion | threshold | measured | result |", "|---|---|---|---|---|"]
    pc1 = r1["per_county"]
    q("C0.v3.R1.decomposition", PRE, {"input": rlg.L0_ALL + [PRE, FULL, L1],
                                      "validator": [PRE1, FULL1, L11, PRE2]},
      "R1 pass; v3 vs v2 differing pump-months " + ", ".join(
          f"{t} {pc1[t]['preorigin']['v3_vs_v2_pump_months_differing']}" for t in C)
      + "; filled/paired pumps " + ", ".join(
          f"{t} {pc1[t]['preorigin']['n_filled_pumps']}/{pc1[t]['preorigin']['n_paired_pumps']}"
          for t in C))
    out.append("| R1 | v3 = independent recomputation from L0; v3 = v1 outside the 2022-12 of "
               "paired pumps (pre-origin); flag-4 rule, values and sd; full has no flag 4; L1 "
               "columns | 0 violations; 1e-6 kWh + 1e-9 rel; fill parameters 1e-9 | 0 "
               "violations in every county and both variants; fill parameters max "
               f"{max(pc1[t]['fill_params_max_abs_diff_vs_product'] for t in C):.1e}; filled "
               "pumps " + ", ".join(f"{pc1[t]['preorigin']['n_filled_pumps']:,}" for t in C)
               + "; v3 differs from v2 in "
               + " + ".join(f"{pc1[t]['preorigin']['v3_vs_v2_pump_months_differing']}"
                            for t in C) + " pump-months, all 2022-12 | pass |")
    pc2 = r2["per_county"]
    for t in C:
        q(f"C0.v3.R2.invariance.{t}", PRE, {"input": rlg.l0(t) + [PRE, L1],
                                           "validator": [PRE1]},
          f"perturbed {pc2[t]['perturbation']['pump_months_changed']} pump-months: "
          f"{pc2[t]['violations']} violations; v1 control {pc2[t]['control_v1_builder']}")
    out.append("| R2 | pre-origin v3 and L1 columns rebuilt from an L0 with every month >= "
               "2023-01 overwritten and the 2022-12 of paired meters x U(0,5) equal the "
               "delivered v3; v1's builder must change (positive control) | 0 violations; "
               "control detects the leak in Yunlin and Changhua | 0 violations ("
               + ", ".join(f"{pc2[t]['perturbation']['pump_months_changed']:,}" for t in C)
               + " pump-months perturbed); v1 control: kept-uid differences Yunlin "
               f"{pc2['yunlin']['control_v1_builder']['kept_uid_symdiff']}, Changhua "
               f"{pc2['changhua']['control_v1_builder']['kept_uid_symdiff']}; Chiayi "
               f"{pc2['chiayi']['control_v1_builder']['pump_months_changed']} pump-months "
               "changed (2022-12) | pass |")
    y3 = r3["v3"]
    rel_dec = y3["delta_fan_total_2022_12_kwh"] / y3["model_fan_total_2022_12_kwh"]
    q("C0.v3.R3.yunlin_vs_model", PRE, {"input": rlg.l0("yunlin") + [PRE], "validator": [PKL]},
      f"v3: max monthly rel 2012-01..2022-11 {y3['max_rel_month_2012_01_2022_11']:.3g} "
      f"({y3['worst_month_2012_01_2022_11']}), sum|d| {y3['sum_abs_kwh_2012_01_2022_11']:.0f} "
      f"kWh, 2022-12 {rel_dec:+.4f}, GWh/yr {y3['fan_total_GWh_per_yr_2012_2022']:.3f}")
    out.append("| R3 | Yunlin county axis vs the calibrated model's cached forcing (task d "
               "restated) | none (reported) | 2012-01..2022-11: max monthly "
               f"{y3['max_rel_month_2012_01_2022_11']:.1e} ({y3['worst_month_2012_01_2022_11']}),"
               f" sum \\|d\\| {y3['sum_abs_kwh_2012_01_2022_11'] / 1e3:.0f} MWh; 2022-12 "
               f"{pct(rel_dec, 2)} (v1: 1.7e-5; 2022-12 +40 kWh) | reported |")
    r4 = e["R4"]
    q("C0.v3.R4.violations", PRE, {"input": rlg.L0_ALL + [PRE, FULL, L1]},
      f"pre {r4['preorigin']['violations']}, full {r4['full']['violations']}, pass {r4['pass']}")
    out.append("| R4 | masks and identities (pre-install zeros never imputed, NaN <=> flag 8, "
               "flag 4 only on paired 2022-12, cells = sum of pumps, totals, long table, sd "
               "recomputed with the meter class, missing_expected recomputed, 2026-07 masks, "
               "kept sets) | 0 violations; 1e-6 kWh | 0 (pre-origin), 0 (full); sd "
               "recomputation max "
               f"{r4['preorigin']['grids']['sd_recomputed_cells_by_class']['max_abs']:.1e} kWh "
               "| pass |")
    q("C0.v3.R5.privacy", PRE, {"input": [PRE, FULL, L1]},
      f"{len(e['R5']['violations'])} violations over {e['R5']['n_files']} files")
    out.append(f"| R5 | privacy scan of every v3 output | 0 | 0 over {e['R5']['n_files']} "
               f"files ({e['R5']['n_raw_meter_numbers']:,} raw meter numbers) | pass |")
    cnt = e["R6"]["counts"]
    q("C0.v3.R6.counties", PRE, {"input": rlg.L0_ALL + [
        lib.rel(lib.PARQ / "pingtung_census.parquet"), GRID, PRE, L1]}, f"{cnt}")
    out.append("| R6 | counties with pumps in active cells | Yunlin, Changhua > 0; Chiayi 320; "
               "Pingtung 0 | kept in active cells "
               + ", ".join(f"{cnt[t]['kept_pumps_in_active_cells']:,}" for t in C)
               + f"; Chiayi census {cnt['chiayi']['census_pumps_in_active_cells']}, Pingtung "
               f"{cnt['pingtung']['census_pumps_in_active_cells']} | pass |")
    q("C0.v3.T5.violations", PRE, {"input": [PRE, FULL]},
      f"{e['T5']['violations']} violations; class x cell entries at 2022-12 with a paired pump "
      f"{e['T5']['items']['n_cells_dec2022_with_paired_pump']['count']}")
    out.append("| T5 | pre-origin = full up to 2022-11; at 2022-12 identical except where a "
               "paired pump sits | 1e-9 kWh, 0 mismatches | 0; "
               f"{e['T5']['items']['n_cells_dec2022_with_paired_pump']['count']:,} class x cell "
               "entries at 2022-12 hold a paired pump | pass |")
    cc, ci = r7["class_x_cell"], r7["county_in_grid"]
    cy = [r for r in ci["table"] if r["county"] == "chiayi"]
    q("C0.v3.R7.backtest", PRE, {"validator": rlg.L0_ALL, "input": [PRE, L1]},
      f"class x cell coverage {cc['coverage_90']:.4f} (n {cc['n']}), median |err| "
      f"{cc['median_abs_rel_error']:.4f}; county YL+CH {ci['coverage_90_yunlin_changhua']:.3f}, "
      f"median |err| {ci['median_abs_rel_error']:.4f}; Chiayi fill > realised in "
      f"{sum(r['F'] > r['A'] for r in cy)} of {len(cy)} years; label {r7['sd_label']}")
    out.append("| R7 | leave-one-year-out back-test of the fill and its sd, 2012-2021 | "
               "'calibrated' iff class x cell coverage of the 90 % interval in [0.80, 0.97] "
               "and Yunlin+Changhua county coverage >= 0.75 (label only) | class x cell "
               f"{cc['coverage_90']:.3f} (n {cc['n']:,}; median \\|err\\| "
               f"{cc['median_abs_rel_error']:.2f}); county {ci['coverage_90_yunlin_changhua']:.2f}"
               f" (median \\|err\\| {ci['median_abs_rel_error']:.3f}); Chiayi's fill exceeded "
               f"the realised half in {sum(r['F'] > r['A'] for r in cy)} of {len(cy)} years | "
               f"**{r7['sd_label']}** |")
    p8 = r8["per_county"]
    for t in C:
        q(f"C0.v3.R8.fill_vs_actual.{t}", PRE, {"input": [PRE], "validator": rlg.l0(t) + [PRE1]},
          f"fill - actual {p8[t]['fill_minus_actual_rel']:.4f}; rule sensitivity "
          f"{p8[t]['rule_sensitivity']:.4f}, precision {p8[t]['rule_precision']:.4f}; untreated "
          f"{p8[t]['untreated_equal_halves_meters']} meters, share "
          f"{p8[t]['untreated_share_of_dec2022_in_grid_meter_level']:.5f}")
    out.append("| R8 | after the fact (January-2023 data): fill vs the realised December "
               "half; capture of equal-half meters | none (reported) | fill - realised "
               + ", ".join(f"{NAME[t]} {p8[t]['fill_minus_actual_rel']:+.1%}" for t in C)
               + "; rule sensitivity "
               + "/".join(f"{p8[t]['rule_sensitivity']:.3f}" for t in C) + ", precision "
               + "/".join(f"{p8[t]['rule_precision']:.3f}" for t in C) + " | reported |")
    p9 = r9["per_county"]
    for t in C:
        q(f"C0.v3.R9.snapshot.{t}", PRE, {"input": rlg.l0(t) + [PRE]},
          f"pumps installed after 2022-12: {p9[t]['pumps_install_after_2022_12']} (in grid "
          f"{p9[t]['pumps_install_after_2022_12_in_grid']}); rel change 2012-2022 "
          f"{p9[t]['rel_change_2012_2022']}; max monthly {p9[t]['max_rel_month_2012_2022']}")
    out.append("| R9 | sensitivity of the census-attribute exemption: rebuild without pumps "
               "installed after 2022-12 | none (reported) | "
               + ", ".join(f"{NAME[t]} {p9[t]['pumps_install_after_2022_12']} pumps, "
                           f"{p9[t]['rel_change_2012_2022']:+.1e}" for t in C)
               + " of 2012-2022 in-grid energy | reported |")
    for v, P in (("preorigin", PRE), ("full", FULL)):
        for t in C:
            q(f"C0.v3.R10.{v}.energy.{t}", P, {"input": rlg.l0(t) + [GRID]},
              f"in-grid {r10[v][t]['in_grid_GWh_per_yr_2012_2022']:.4f} GWh/yr 2012-2022, "
              f"domestic {r10[v][t]['domestic_share_2012_2022']:.4f}")
    out.append("| R10 | descriptive | none | in-grid GWh/yr 2012-2022, pre-origin "
               + ", ".join(f"{r10['preorigin'][t]['in_grid_GWh_per_yr_2012_2022']:.2f}"
                           for t in C) + " (full "
               + ", ".join(f"{r10['full'][t]['in_grid_GWh_per_yr_2012_2022']:.2f}" for t in C)
               + "); domestic share "
               + ", ".join(pct(r10['preorigin'][t]['domestic_share_2012_2022'])
                           for t in ("yunlin", "changhua")) + " | reported |")
    gates = e["summary"]["gates"]
    q("C0.v3.gates", PRE, {"input": rlg.L0_ALL + [PRE, FULL, L1]},
      "; ".join(f"{k.split('/')[-1]} {v['passed']}" for k, v in gates.items()))
    out.append("| gates | preorigin (R1 pre, R2, R4 pre, R5, R6, T5); full (R1 full, R4 full, "
               "R5, R6, T5); census_L1 (R2 L1, R4 kept, R5, R6) | conjunction | "
               + "; ".join(f"{k.split('/')[-1]} {'passed' if v['passed'] else 'FAILED'}"
                           for k, v in gates.items()) + " | pass |")
    out.append("")

    # 7.4 exemptions
    for t in C:
        q(f"C0.v3.residual.{t}", PRE, {"input": [PRE, L1], "validator": rlg.l0(t)},
          f"untreated equal-half meters {res[t]['untreated_meters_all']} "
          f"({res[t]['untreated_meters_kept_in_grid']} kept in grid); Dec-2022 in-grid "
          f"{res[t]['dec2022_in_grid_kWh_on_untreated_meters']:.2f} kWh, share "
          f"{res[t]['share_of_dec2022_in_grid']:.5f} of Dec-2022, "
          f"{res[t]['share_of_2012_2022_in_grid']:.2e} of 2012-2022; flips "
          f"{res[t]['kept_flips_kept_to_dropped'] + res[t]['kept_flips_dropped_to_kept']}")
    for name, P in (("preorigin", PRE), ("census_L1", L1)):
        q(f"C0.v3.exemption.census_attributes.{name}", P,
          {"input": [lib.rel(lib.PARQ / f"{t}_census.parquet") for t in C]},
          "exemption (U3) in the v3 manifest: census attribute table of the 2026 delivery, "
          "data_end 2026-07, non_exempt_fit_window_end 2022-12; sensitivity R9 (v3)")
        q(f"C0.v3.exemption.residual_unpaired_dec2022.{name}", P,
          {"input": [lib.rel(lib.PARQ / f"{t}_kwh_long.parquet") for t in C]},
          "exemption (U3) in the v3 manifest: 2022-12 readings of meters not labelled "
          "Dec-Jan-paired whose Dec-2022 equals Jan-2023, data_end 2023-01; sensitivity R8 + "
          "repair3_residual.py")
    for name, P in (("preorigin", PRE2), ("census_L1", L12)):
        q(f"C0.v2.exemption.census_attributes.{name}", P,
          {"input": [lib.rel(lib.PARQ / f"{t}_census.parquet") for t in C]},
          "exemption (U3) named in the v2 manifest (gate failed): census attribute table, "
          "data_end 2026-07; sensitivity R9 (v2)")
    out += ["### 7.4 Exemptions of `preorigin/v3` and `census_L1/v3` (plan 2.3, U3)", "",
            "1. **Census attribute table** (2026 delivery, data to 2026-07). No 2022-12 "
            "snapshot exists. R9 rebuilt each county without the pumps whose census install "
            "date is after 2022-12: "
            + "; ".join(f"{NAME[t]} {p9[t]['pumps_install_after_2022_12']} pumps, 2012-2022 "
                        f"in-grid energy {p9[t]['rel_change_2012_2022']:+.1e} (max monthly "
                        f"{p9[t]['max_rel_month_2012_2022']:.1e})" for t in C) + ".",
            "2. **Residual unpaired December-2022 halves** (data to 2023-01). This exemption "
            "was added in the resume session, after R8 had been read. R8 found meters whose "
            "Dec-2022 and Jan-2023 readings are equal non-zero halves but which the "
            "pre-origin rule does not label paired; their 2012-01..2022-11 history shows no "
            "Dec-Jan pairing. Only the January-2023 reading can identify them, so no "
            "pre-origin rule can fill or mask them. Their 2022-12 value stays as read. "
            "`repair3_residual.py` (after the fact, logged [C0-repair3-residual]) measured:",
            "",
            "   | county | meters (kept in grid) | 2022-12 in-grid kWh | share of in-grid Dec-2022 "
            "| share of 2012-2022 | kept flips if left out of the duty |",
            "   |---|---|---|---|---|---|"]
    for t in C:
        out.append(f"   | {NAME[t]} | {res[t]['untreated_meters_all']:,} "
                   f"({res[t]['untreated_meters_kept_in_grid']:,}) | "
                   f"{res[t]['dec2022_in_grid_kWh_on_untreated_meters']:,.0f} | "
                   f"{pct(res[t]['share_of_dec2022_in_grid'], 2)} | "
                   f"{res[t]['share_of_2012_2022_in_grid']:.1e} | "
                   f"{res[t]['kept_flips_kept_to_dropped'] + res[t]['kept_flips_dropped_to_kept']}"
                   " |")
    out += ["",
            "Both exemptions name the exemptable allowed uses explicitly. Each has a ledger id "
            "(`C0.v3.exemption.<kind>.<product>`) and ledger rows. The guard grants "
            "`forcing_frozen` on `preorigin/v3` only through them; its notes name both "
            "exemptions (guard probe).", ""]

    # 7.5 minor findings
    pa = e["facts"]["phase_agreement_v3_build_info"]
    for t in C:
        q(f"C0.v3.facts.m1_pair_equality.{t}", PRE, {"validator": rlg.l0(t)},
          "M1 Dec_y/Jan_y+1 equal fraction: " + ", ".join(
              f"{k} {v['frac_equal']:.4f}" for k, v in
              fa[t]["m1_dec_jan_pair_equality"].items()))
        q(f"C0.v3.facts.no_coordinates.{t}", L1, {"input": [lib.rel(lib.PARQ /
                                                                     f"{t}_census.parquet")],
                                                 "validator": [lib.rel(lib.PARQ /
                                                                       f"{t}_kwh_long.parquet")]},
          f"census rows without coordinates {fa[t]['rows_without_coordinates']} of "
          f"{fa[t]['census_rows']}; raw kWh share 2012-2022 "
          f"{fa[t]['rows_without_coordinates_raw_kwh_share_2012_2022']:.5f}")
        q(f"C0.v3.facts.phase_agreement.{t}", PRE, {"input": rlg.l0(t) + [PRE, FULL]},
          f"pre bimonthly {pa['preorigin'][t]['phase_agree_frac_bimonthly']:.4f} monthly "
          f"{pa['preorigin'][t]['phase_agree_frac_monthly']:.4f}; full bimonthly "
          f"{pa['full'][t]['phase_agree_frac_bimonthly']:.4f} monthly "
          f"{pa['full'][t]['phase_agree_frac_monthly']:.4f}")
    key = "2021-12/2022-01"
    other = min(v["frac_equal"] for t in C for k, v in fa[t]["m1_dec_jan_pair_equality"].items()
                if k != key)
    man = lib.load_json(lib.REPO / PRE / "MANIFEST.json")
    n_refs = sum(1 for i in man["inputs"] if str(i.get("role", "")).startswith("validation_"))
    out += ["### 7.5 The minor findings", "",
            "1. **Task (d) 'to within rounding'.** v1 met it only with a tolerance set after "
            "T2a (one AMP-absent pump, 6,782 kWh). This is acknowledged. v3 departs from the "
            "model's forcing by design (R3: max monthly "
            f"{y3['max_rel_month_2012_01_2022_11']:.1e} on 2012-01..2022-11, 2022-12 "
            f"{pct(rel_dec, 2)}). The model's own cached forcing carries the same January-2023 "
            "information as v1, so identity with it and U3 cannot both hold. Pipeline "
            "identity is still shown by T1a/T1b/T2a (v1; the cleaning code is unchanged), and "
            "R1 accounts exactly for every v3-v1 difference: kept flips plus the fill.",
            "2. **Flag semantics against allowed uses.** `scoring_target` and `selection` are "
            "removed from `preorigin/v3`, and the guard refuses both. The pump-level method "
            "codes still separate pre-install zeros, even-split halves and the rest. The long "
            "table's `method` stays 'observed' (stated in the card).",
            "3. **L0 anomaly.** Equal halves in the M1 pair 2021-12/2022-01: "
            + ", ".join(f"{pct(fa[t]['m1_dec_jan_pair_equality'][key]['frac_equal'])} "
                        f"({NAME[t]})" for t in C)
            + f", against >= {pct(other)} in every other year 2012-2024. Pair totals are "
            "unaffected, and the fill ratio uses pair sums. Stated in the product cards.",
            "4. **Billing-code agreement figures.** Round 1 quoted 0.96-0.98 / 0.93-0.95; "
            "corrected in section 1. Bimonthly agreement is "
            + ", ".join(f"{pa['preorigin'][t]['phase_agree_frac_bimonthly']:.3f}" for t in C)
            + " (pre-origin; Chiayi is lowest). Monthly agreement is "
            + ", ".join(f"{pa['preorigin'][t]['phase_agree_frac_monthly']:.3f}" for t in C)
            + ".",
            "5. **checks.py edited after T1a-T9 were scored** (the T3 crash fix). The file "
            "that produced the round-1 gating evidence is not preserved; this stands. In the "
            "repair round every check JSON records the checks-code sha256 at scoring time, "
            "and every superseded checks file is archived by sha256 under "
            "`checks/code_archive/`. All v3 tests ran on one archived version, which the "
            "resume check confirmed.",
            f"6. **Validation inputs outside the checked hashes.** The v3 manifests list "
            f"{n_refs} validation references (the AMP_V2 census and kWh, the cached-forcing "
            "pickle, T2a.json, and the v1 and v2 product files) as `inputs` with "
            "role `validation_reference`. The checker now verifies them.",
            "7. **Census attributes from the 2026 delivery.** Declared as exemption 1, with "
            "R9 as its sensitivity.",
            "8. **sd NaN everywhere.** The fill now has an sd, labelled calibrated by R7, and "
            "post-install missing readings have `missing_expected_kwh`. Observed values still "
            "carry no sd, because the TPC reading error is unknown: partly addressed.",
            "9. **SOURCE.json equality claim.** The claim of exact equality with AMP_V2 is "
            "contradicted by one pump (223 pump-months). This is for the owner of the L0 "
            "SOURCE.json; C0 does not edit L0.",
            "10. **Chiayi rows without coordinates.** "
            + ", ".join(f"{NAME[t]} {fa[t]['rows_without_coordinates']} of "
                        f"{fa[t]['census_rows']:,} rows "
                        f"({pct(fa[t]['rows_without_coordinates_raw_kwh_share_2012_2022'], 3)} "
                        "of the county's raw 2012-2022 kWh)" for t in C)
            + " have no cell. They are left out of the cell arrays, the model's convention. "
            "Stated in the cards.",
            "11. **Build cache `pump_electricity/_build_c0/`** (198 MB; per-pump arrays with "
            "well numbers; no manifest). It is unchanged, gitignored and is not a product. "
            "The repair builds wrote their intermediates elsewhere and the products hold "
            "their own copies, so the cache can be deleted by the owner.", ""]

    # 7.6 deviations
    nl = len(e["looks"])
    q("C0.v3.guard_probe", PRE, {"input": [PRE, FULL, L1, PRE2, PRE1]},
      "; ".join(f"{c['product']}/{c['version']} {c['use']}: {c['result'][:30]}"
                for c in e["probe"]["cases"]))
    q("C0.v3.manifests_check", PRE, {"input": [PRE, FULL, L1, PRE2, FULL2, L12, PRE1, FULL1,
                                               L11]},
      "; ".join(f"{m['path'].split('pump_electricity/')[-1]} {m['status']} {m['n_hashed']}"
                for m in e["probe"]["manifests"]))
    rs = e["resume"]
    q("C0.v3.resume_check", PRE, {"input": [PRE, FULL, L1]},
      f"prereg allowed {rs['prereg_status']['allowed']}, checks ok {rs['checks_ok']}, "
      f"published manifests ok {rs['published_manifests_ok']}, c0 refused "
      f"{len(rs['audit']['c0_refused'])}")
    out += ["### 7.6 Deviations in the repair round", "",
            "- **Order of work.** The v2 and v3 products were built before their "
            "pre-registrations and are pinned in them by sha256, as in round 1. The tests "
            "are construction, invariance and consistency checks, not out-of-sample tests. "
            "v3 was designed after the v2 R4 and T5 failures were known.",
            "- **v2 published with a failed gate.** Version directories are immutable. The "
            "v2 manifests record gate False, and the guard refuses v2 for every use that "
            "needs a gate.",
            "- **Task (d) literal** is not met by v3 (section 7.5, item 1).",
            "- **Residual exemption added after reading R8.** It was not pre-registered. Its "
            "sensitivity (`repair3_residual.py`) is an unregistered after-the-fact "
            "computation, logged before it ran.",
            "- **Resume after a usage-limit cut.** The interrupted session scored every v3 "
            "test (08:34-08:46Z on 2026-10-13) but stopped before the manifests, ledger and "
            "report. This session rebuilt and re-scored nothing. It logged a resume look "
            "([C0-repair3-resume-1]) and ran a read-only integrity check (prereg3 allows all "
            f"11 tests: {rs['prereg_status']['allowed']}; every check file matches the "
            f"registration and the checks code: {rs['checks_ok']}; every v3 file pinned and "
            "unchanged; v1/v2 manifests re-hash). It then added the residual sensitivity, the "
            "report facts and a guard probe, and finalised v3 with `repair3_finalize.py`. "
            "That file is rewritten from the interrupted session's unrun draft, so the v3 "
            "texts no longer say v2 and the second exemption is declared.",
            f"- **Look log.** {nl} C0 repair-round records: seeds, design looks, scores, "
            "diagnostics and resume looks.",
            "- **Ledger.** @@LEDGER@@",
            "- No Claude API (0 tokens) and no GPU. Nothing was written under `hydrophysics/`, "
            "`tests/` or L0, and nothing was staged in git.", ""]

    # 7.7 downstream
    fill_tot = sum(p8[t]["fill_in_grid_kWh"] for t in C)
    act_tot = sum(p8[t]["v1_value_same_pumps_kWh"] for t in C)
    q("C0.v3.R8.fan_total", PRE, {"input": [PRE], "validator": rlg.L0_ALL + [PRE1]},
      f"in-grid fill {fill_tot:.0f} kWh vs realised {act_tot:.0f} kWh; sd upper "
      f"{r10['preorigin']['dec2022_sd_fan_total_upper_kWh']:.0f} kWh")
    out += ["### 7.7 Issues for downstream (repair round)", "",
            "- **Use v3.** `preorigin/v1` still opens for `forcing_frozen` (guard probe), "
            "because its manifest cannot be changed, but it carries the January-2023 leak. "
            "`preorigin/v2` is refused (gate failed).",
            "- **The calibrated model's own forcing has the same leak.** The cached "
            "`E_by_class` in `stage3_datum_gate_cwa/inputs.local.pkl` equals v1 for Yunlin "
            f"(T1a). Its 2022-12 carries the paired meters' half bills "
            f"({pct(p8['yunlin']['v1_paired_share_of_dec2022'])} of Yunlin's in-grid "
            "December-2022), and its kept set saw them. Model integration (after the code "
            "freeze) should take `preorigin/v3` and recalibrate. The difference from the "
            "current forcing is "
            f"{y3['max_rel_month_2012_01_2022_11']:.1e} monthly before 2022-12 and "
            f"{pct(rel_dec, 2)} in 2022-12.",
            "- **`pump_electricity/model_census/v1`** appeared in this product namespace on "
            "2026-10-14. It was built by `dataprep_staging/i1/build_model_census.py`, not by "
            "C0. It imports `c0_pump_census/lib.py`, so that file must stay unchanged. Its "
            "kWh window runs to 2023-01 and it allows `forcing_frozen`. If it follows the "
            "model's convention, its 2022-12 carries the same Dec-Jan halves; its owner "
            "should check this against v3's fill.",
            "- **Fill quality.** At county level the median back-test error is "
            f"{pct(ci['median_abs_rel_error'])}. After the fact the in-grid fill fell short of "
            "the realised December-2022 halves by "
            f"{(fill_tot - act_tot) / 1e6:+.2f} GWh in all ("
            + ", ".join(f"{NAME[t]} {p8[t]['fill_minus_actual_rel']:+.1%}" for t in C)
            + "), smaller than the upper bound of the fill's fan-total sd, "
            f"{r10['preorigin']['dec2022_sd_fan_total_upper_kWh'] / 1e6:.2f} GWh. Chiayi's "
            "fill is biased high "
            f"({sum(r['F'] > r['A'] for r in cy)} of {len(cy)} back-test years; R8 "
            f"{p8['chiayi']['fill_minus_actual_rel']:+.0%}), but Chiayi is "
            f"{pct(r10['preorigin']['chiayi']['in_grid_GWh_per_yr_2012_2022'] / sum(r10['preorigin'][t]['in_grid_GWh_per_yr_2012_2022'] for t in C), 2)} "
            "of in-grid energy.",
            "- **Flag 4 in forcing_frozen.** The guard keeps flags 0-5 for `forcing_frozen`, "
            "so the fill enters. A consumer wanting only observations can drop flag 4, and "
            "the 2022-12 cells holding a paired pump then become partial sums.",
            "- **Recalibration needed anyway.** The v3 kept set differs from v1's (the "
            "model's, for Yunlin) by "
            + ", ".join(f"{r1['per_county'][t]['flips_in_meters'] + r1['per_county'][t]['flips_out_meters']}"
                        f" {NAME[t]}" for t in C)
            + " meters, and Changhua is new to the model.", ""]
    return "\n".join(out)


HEADER = """# WP C0: cleaned pump-electricity census (Yunlin, Changhua, in-grid Chiayi)

**Status: passed after the repair round.** The current products are `pump_electricity/{{census_L1,preorigin,full}}/v3`. They pass every gate of the pre-registration `C0.census_product.v3` (sha256 `{sha8}…`).

The independent verifier found one critical issue: a U3/U4 breach in `preorigin/v1`, whose 2022-12 values carried January-2023 electricity. v3 fills December 2022 of the affected meters from pre-origin data and takes the January-2023 information out of the kept-set statistic. The registered perturbation test R2 shows the pre-origin products no longer depend on any post-origin reading. What post-origin information remains is declared as two manifest exemptions, each with a sensitivity run. Task (d) taken literally ("to within rounding" of the model's forcing) is no longer met: v3 departs from the leaky model forcing by design (section 7.5).

**Do not use v1 or v2 for any pre-origin purpose.** v2 failed its registered gate. v1's manifest still grants forcing_frozen, but it carries the leak.

Section 7 (Repair round) describes the repair. Sections 1-6 describe round 1 (the v1 products) and are kept for the record; corrections are marked "(corrected in the repair round)".

Machine-readable version: `results/dataprep/reports/C0.json`. Evidence: `results/dataprep/reports/C0_evidence/` (round 1), `.../C0_evidence/repair/` (v2) and `.../C0_evidence/repair3/` (v3).
"""


def update_md(e: dict, sec7: str) -> None:
    txt = MD.read_text(encoding="utf-8")
    k = txt.find("\n## 7. Repair round")
    if k >= 0:
        txt = txt[:k + 1]
    i = txt.index("## 1. What was built")
    body = txt[i:]
    old39 = ("Agreement with the billing code is 0.96-0.98 for bimonthly meters and 0.93-0.95 "
             "for monthly ones.")
    pa = e["facts"]["phase_agreement_v3_build_info"]["preorigin"]
    new39 = ("Agreement with the billing code is "
             + ", ".join(f"{pa[t]['phase_agree_frac_bimonthly']:.3f}" for t in C)
             + " (Yunlin, Changhua, Chiayi) for bimonthly meters and "
             + ", ".join(f"{pa[t]['phase_agree_frac_monthly']:.3f}" for t in C)
             + " for monthly ones (pre-origin; corrected in the repair round, round 1 quoted "
               "0.96-0.98 and 0.93-0.95).")
    if old39 in body:
        body = body.replace(old39, new39)
    old44 = "- **sd** is NaN everywhere, because every value is an observation or a structural zero."
    new44 = (old44 + " (v1 only; v3 adds an sd for the December-2022 fill, section 7.)")
    if old44 in body and new44 not in body:
        body = body.replace(old44, new44)
    out = HEADER.format(sha8=e["prereg_sha"][:8]) + "\n" + body.rstrip() + "\n\n" + sec7
    MD.write_text(out, encoding="utf-8")


# ----------------------------------------------------------------------------- json
def summary(e: dict) -> str:
    r10, r8, res = e["R10"], e["R8"]["per_county"], e["resid"]["per_county"]
    r3 = e["R3"]["v3"]
    rel_dec = r3["delta_fan_total_2022_12_kwh"] / r3["model_fan_total_2022_12_kwh"]
    gw = r10["preorigin"]
    return (
        "C0 delivers the cleaned Taiwan Power pump-electricity census for every county with "
        "pumps in the twin's active cells (Yunlin, Changhua, the 320 in-grid Chiayi pumps). It "
        "is built from the new L0 parquet only and cleaned with "
        "hydrophysics.twin.pumping.clean_census as the model cleans Yunlin. In the repair "
        "round the verifier's critical finding is fixed. preorigin/v1 carried January-2023 "
        "electricity: the 2022-12 value of each Dec-Jan-paired (phase-B) meter is half of a "
        "bill read in 2023, " + ", ".join(
            f"{r8[t]['v1_paired_share_of_dec2022']:.0%}" for t in C)
        + " of the in-grid Dec-2022 energy (Yunlin, Changhua, Chiayi), and its duty statistic "
        "saw it. The current products are census_L1/v3, preorigin/v3 and full/v3. The pairing "
        "is now detected from pre-origin data only. The paired meters' Dec-2022 value leaves "
        "the duty statistic, which the L1 table and both variants share. In the pre-origin "
        "variant that value becomes a flagged fill (flag 4) from the Oct-Nov 2022 base and a "
        "2012-2021 ratio, with an sd that the back-test R7 labels calibrated. All six gating "
        "tests of C0.census_product.v3 pass, as do the three product gates. R2 perturbs every "
        "post-origin month and every paired 2022-12 reading and finds 0 differences in the "
        "pre-origin products, while v1's builder fails the same test. Two exemptions with "
        "sensitivities are declared: the 2026 census attribute table (R9, <= 3.7e-4 of "
        "2012-2022 energy) and the residual Dec-2022 halves the pre-origin rule cannot see ("
        + ", ".join(f"{res[t]['share_of_dec2022_in_grid']:.3%}" for t in C)
        + " of Dec-2022, no kept-set flip). v2 failed its gate (an R4 check defect and a T5 "
        "product defect) and is superseded. Task (d) taken literally is not met by v3, by "
        f"design: max monthly {r3['max_rel_month_2012_01_2022_11']:.1e} before 2022-12 and "
        f"{rel_dec:+.2%} in 2022-12 against the model's leaky cached forcing; pipeline "
        "identity rests on round-1 T1a-T2a. In-grid 2012-2022 energy is "
        + ", ".join(f"{gw[t]['in_grid_GWh_per_yr_2012_2022']:.2f}" for t in C)
        + " GWh/yr. The session resumed after a usage-limit cut. It re-verified the "
        "interrupted session's v3 scores read-only, then finalised the manifests, ledger and "
        "report. Nothing was rebuilt or re-scored; no GPU and no Claude API were used.")


def deviations(e: dict) -> list[str]:
    r3 = e["R3"]["v3"]
    rel_dec = r3["delta_fan_total_2022_12_kwh"] / r3["model_fan_total_2022_12_kwh"]
    res = e["resid"]["per_county"]
    return [
        "Task (d) taken literally ('to within rounding' of the model's cached Yunlin forcing) "
        "is not met by preorigin/v3, by design. Max monthly fan-total difference is "
        f"{r3['max_rel_month_2012_01_2022_11']:.1e} on 2012-01..2022-11 (kept-set flips) and "
        f"{rel_dec:+.2%} in 2022-12 (the fill). U3 takes precedence because the model's "
        "cached forcing itself carries the January-2023 information. Pipeline identity rests "
        "on round-1 T1a/T1b/T2a (cleaning code unchanged), and R1 accounts for every v3-v1 "
        "difference.",
        "preorigin/v3 and census_L1/v3 declare two exemptions (plan 2.3). The first is the "
        "2026 census attribute table (sensitivity: registered R9). The second is the residual "
        "Dec-2022 halves the pre-origin rule cannot identify: " + ", ".join(
            f"{t} {res[t]['share_of_dec2022_in_grid']:.2%}" for t in C)
        + " of the in-grid Dec-2022, 0 kept flips. The second was added in the resume session "
        "after R8 had been read; it was not pre-registered, and its sensitivity is an "
        "unregistered after-the-fact computation (repair3_residual.py, logged before it ran).",
        "The pre-origin forcing now holds an imputed value (flag 4) in 2022-12 for "
        "Dec-Jan-paired meters. It is a forcing fill with a back-tested sd, never a scoring "
        "target (scoring_target and selection are not allowed uses of preorigin/v3).",
        "v2 was published with a failed registered gate (an R4 check defect and a T5 product "
        "defect), because version directories are immutable. v1 and v2 are superseded, but "
        "v1's manifest still grants forcing_frozen (the guard has no supersession mechanism).",
        "Products of all three versions were built before their pre-registrations and are "
        "pinned in them by sha256. The registered tests are construction, invariance and "
        "consistency checks, not out-of-sample tests. v3's fill-base rule and the corrected "
        "R4/T5 checks were decided after the v2 R4/T5 failures were known (logged "
        "[C0-repair-v3-decision], before any v2 R3/R7/R8/R10 result was read).",
        "Round 1, unchanged: the pre-install coding is not harmonised across counties (the "
        "duty statistic sees the L0 readings as delivered; Changhua zero-coding would add 64 "
        "meters, +0.35 %).",
        "Round 1, unchanged: the TPC/P1 validation (T3 reported, T4 descriptive) is after the "
        "fact, because P1's products fail their gates. It was not rerun on v3. The v3 full "
        "variant differs from v1 only through the kept-set flips (R1), with in-grid "
        "2012-2022 energy " + ", ".join(
            f"{e['R10']['full'][t]['in_grid_GWh_per_yr_2012_2022']:.2f}" for t in C)
        + " GWh/yr against v1's 188.46, 187.41 and 0.63.",
        "Round 1, unchanged: TPC's even split of bimonthly bills is kept; P2 builds on C0. The "
        "full variant ships an even-split alternative from 2025-07.",
        "sd exists only for the December-2022 fill. Observed values keep sd NaN, because the "
        "TPC reading error is not modelled. Post-install missing readings count as zero, with "
        "the indicator missing_expected_kwh.",
        "Round-1 provenance gap: checks.py was edited after T1a-T9 were scored (T3 fix), and "
        "the scoring version is not preserved. Repair-round checks record and archive their "
        "code by sha256.",
        "Resumed WP. The interrupted session built, registered and scored v3. This session "
        "rebuilt and re-scored nothing: it ran a read-only integrity check, added the "
        "residual sensitivity, report facts and a guard probe, and rewrote the unrun draft "
        "repair3_finalize.py before using it.",
    ]


def issues(e: dict) -> list[str]:
    r3 = e["R3"]["v3"]
    rel_dec = r3["delta_fan_total_2022_12_kwh"] / r3["model_fan_total_2022_12_kwh"]
    p8 = e["R8"]["per_county"]
    r10 = e["R10"]["preorigin"]
    return [
        "Use pump_electricity/{census_L1,preorigin,full}/v3. preorigin/v1 still opens for "
        "forcing_frozen (its manifest cannot change) but carries the January-2023 leak; v2 is "
        "refused (gate failed).",
        "Model integration (after the code freeze): the calibrated model's cached forcing "
        "(E_by_class, inputs.local.pkl) equals v1 for Yunlin, so its 2022-12 carries the "
        f"paired meters' half bills ({p8['yunlin']['v1_paired_share_of_dec2022']:.1%} of "
        "Yunlin's in-grid Dec-2022), and its kept set saw them. Recalibrate on preorigin/v3 "
        f"(U1). The difference is {r3['max_rel_month_2012_01_2022_11']:.1e} monthly before "
        f"2022-12 and {rel_dec:+.2%} in 2022-12. Adding Changhua adds "
        f"{r10['changhua']['in_grid_GWh_per_yr_2012_2022']:.1f} GWh/yr to "
        f"{r10['yunlin']['in_grid_GWh_per_yr_2012_2022']:.1f}; the Yunlin-only K_pump, theta "
        "and column do not transfer. C0 does not depend on theta.",
        "pump_electricity/model_census/v1 appeared in the C0 product namespace on 2026-10-14. "
        "It was built by dataprep_staging/i1/build_model_census.py, not by C0. It imports "
        "c0_pump_census/lib.py (keep it unchanged), reads kWh to 2023-01 and allows "
        "forcing_frozen. Its owner should check its Dec-2022 convention against v3's fill.",
        "Changhua domestic class: "
        f"{r10['changhua']['domestic_share_2012_2022']:.1%} of in-grid energy (Yunlin "
        f"{r10['yunlin']['domestic_share_2012_2022']:.1%}). It may carry household load; "
        "use cells_total.npz value_nodomestic as the sensitivity member.",
        "Flags: guard.read_table for forcing_frozen keeps flags 0-5, so the Dec-2022 fill "
        "(flag 4) enters and flag-9 rows are dropped. npz users must apply flags themselves. "
        "Dropping flag 4 leaves partial 2022-12 sums where a paired pump sits.",
        "Fill quality: the R7 county median error is "
        f"{e['R7']['county_in_grid']['median_abs_rel_error']:.1%}. After the fact the fill is "
        + ", ".join(f"{t} {p8[t]['fill_minus_actual_rel']:+.1%}" for t in C)
        + " against the realised Dec-2022 halves. Chiayi's fill is biased high (all 10 "
        "back-test years), but Chiayi is 0.17 % of in-grid energy.",
        "A pre-install zero-coding member (+64 Changhua meters, +0.35 %) would need a new "
        "version (v4).",
        "Pre-origin termination labels are provisional for runs starting 2021-2022 "
        "(labels only; values are observed zeros).",
        "Recorded-forcing arm (full/v3): from 2025-07, 12-33 % of bimonthly pairs post the "
        "whole bill to the second month; an even-split alternative is shipped. 2026-07 is "
        "masked (flag 8/9).",
        "M1 pair 2021-12/2022-01: only 3.6-6.3 % of pairs are equal halves (>= 99.6 % in other "
        "years). The monthly split of that pair is TPC's uneven split; totals are unaffected.",
        "TPC 農林漁牧 coverage is about 0.4, not stationary and county-dependent. Do not scale "
        "Changhua pumping by TPC. The census is frozen at mid-2016 installs; check "
        "billing-group completeness before reading a 2025-26 decline as less pumping.",
        "Privacy: census_L1 has pump coordinates and well numbers, and the per-pump arrays "
        "have well numbers (gitignored, personal_data declared). The build cache "
        "pump_electricity/_build_c0 (198 MB, no manifest) also holds well numbers, and its "
        "owner may delete it.",
        "Provenance: the TPC SOURCE.json uses files[].path (A0fix; owner edit), and its claim "
        "of exact AMP_V2 equality is contradicted by one irrigation pump (223 pump-months). "
        "ledger/data_roles.csv was not reassembled (shared file).",
    ]


def update_json(e: dict) -> dict:
    old = lib.load_json(JS)
    r1 = old.get("round_1") or {k: old[k] for k in ("status", "summary", "prereg", "acceptance",
                                                   "products", "deviations_from_plan",
                                                   "issues_for_downstream", "resume",
                                                   "key_numbers") if k in old}
    r3, r7, r8, r10 = e["R3"]["v3"], e["R7"], e["R8"]["per_county"], e["R10"]
    rel_dec = r3["delta_fan_total_2022_12_kwh"] / r3["model_fan_total_2022_12_kwh"]
    res = e["resid"]["per_county"]
    gates = e["summary"]["gates"]
    acc = [
        {"criterion": "(d) pipeline identity: the C0 pipeline reproduces the model's cleaned "
                      "Yunlin forcing on the model's inputs and rules (round-1 T1a, T1b, T2a; "
                      "cleaning code unchanged in v3)",
         "threshold": "every element |d| <= 1e-6 kWh + 1e-9 |ref|",
         "measured": "T1a 0.0 kWh, T1b/T2a 1.5e-10 kWh (C0.census_product.v1)",
         "passed": True},
        {"criterion": "(d) literal: delivered preorigin/v3 Yunlin part equals the model's "
                      "cached forcing 2012-2022 'to within rounding' (R3, reported)",
         "threshold": "within rounding (task); R3 registered as reported-only",
         "measured": f"max monthly {r3['max_rel_month_2012_01_2022_11']:.2e} "
                     f"({r3['worst_month_2012_01_2022_11']}) on 2012-01..2022-11, sum |d| "
                     f"{r3['sum_abs_kwh_2012_01_2022_11']:.0f} kWh; 2022-12 {rel_dec:+.4f}; "
                     "every difference accounted for by R1 (kept flips, fill) plus the "
                     "AMP-absent pump; not met by design (U3 precedes: the model's forcing "
                     "carries the January-2023 information)",
         "passed": False},
    ]
    tests = [("R1", "R1 v3 = independent recomputation; v3 = v1 outside paired 2022-12; fill "
                    "values/sd; L1 columns", "0 violations; 1e-6 kWh + 1e-9 rel",
              "0 violations (3 counties, both variants)"),
             ("R2", "R2 no post-origin dependency (perturbed L0) + positive control on v1",
              "0 violations; control detects the leak in Yunlin and Changhua",
              "0 violations; control detects the leak (Yunlin 9, Changhua 49 kept-uid "
              "differences)"),
             ("R4", "R4 masks and internal consistency (incl. sd recomputation)",
              "0 violations; 1e-6 kWh", "0 / 0 (pre-origin / full)"),
             ("R5", "R5 privacy", "0", f"0 over {e['R5']['n_files']} files"),
             ("R6", "R6 counties with pumps in active cells", "YL, CH > 0; CY 320; PT 0",
              "kept in active cells 114,060 / 113,108 / 316; CY census 320; PT 0"),
             ("T5", "T5 pre-origin = full to 2022-11; 2022-12 identical where no paired pump",
              "1e-9 kWh, 0 mismatches", "0 violations")]
    for k, crit, thr, meas in tests:
        acc.append({"criterion": f"C0.v3.{crit}", "threshold": thr, "measured": meas,
                    "passed": bool(e["summary"]["tests"][k])})
    acc += [
        {"criterion": "C0.v3.R7 fill sd back-test (label only, plan 4.7)",
         "threshold": "calibrated iff class x cell coverage in [0.80, 0.97] and YL+CH county "
                      "coverage >= 0.75",
         "measured": f"class x cell {r7['class_x_cell']['coverage_90']:.3f}, county "
                     f"{r7['county_in_grid']['coverage_90_yunlin_changhua']:.2f}: "
                     f"{r7['sd_label']}", "passed": r7["sd_label"] == "calibrated"},
        {"criterion": "C0.v3.R8 after the fact: fill vs realised Dec-2022 half (reported)",
         "threshold": "none (reported)",
         "measured": ", ".join(f"{t} {r8[t]['fill_minus_actual_rel']:+.3f}" for t in C),
         "passed": True},
        {"criterion": "C0.v3.R9 census-attribute exemption sensitivity (reported)",
         "threshold": "none (reported)",
         "measured": ", ".join(f"{t} {e['R9']['per_county'][t]['rel_change_2012_2022']:+.1e}"
                               for t in C), "passed": True},
    ]
    for g, v in gates.items():
        acc.append({"criterion": f"gate {g}/v3", "threshold": "conjunction of its registered "
                                                             "tests (C0.census_product.v3)",
                    "measured": ", ".join(f"{k}={x}" for k, x in v["tests"].items()),
                    "passed": bool(v["passed"])})
    acc += [
        {"criterion": "verifier critical issue: no post-origin information in the pre-origin "
                      "variant without a declared exemption (U3/U4, plan 2.3)",
         "threshold": "R2 invariance + every residual declared as an exemption with ledger id "
                      "and sensitivity",
         "measured": "R2 0 violations; 2 exemptions (census attributes R9; residual unpaired "
                     "Dec-2022: " + ", ".join(
                         f"{t} {res[t]['share_of_dec2022_in_grid']:.2%} of Dec-2022, 0 flips"
                         for t in C) + ")", "passed": True},
        {"criterion": "v2 gates (C0.census_product.v2)", "threshold": "conjunction",
         "measured": "preorigin False (R4 check defect 1,346 sd violations; T5 product defect "
                     "1 entry 3.5 kWh), full False (T5), census_L1 True; superseded by v3",
         "passed": False},
        {"criterion": "resume integrity (2026-10-14, read-only): prereg3 allows the v3 tests, "
                      "check files match registration and checks code, v3 files pinned, "
                      "v1/v2 manifests re-hash",
         "threshold": "all true",
         "measured": f"allowed {e['resume']['prereg_status']['allowed']}, checks "
                     f"{e['resume']['checks_ok']}, published manifests "
                     f"{e['resume']['published_manifests_ok']}",
         "passed": bool(e["resume"]["checks_ok"] and e["resume"]["prereg_status"]["allowed"]
                        and e["resume"]["published_manifests_ok"])},
        {"criterion": "manifest checker on all pump_electricity manifests (incl. v3)",
         "threshold": "status ok",
         "measured": ", ".join(f"{m['path'].split('pump_electricity/')[-1].replace('/MANIFEST.json', '')}"
                               f" {m['status']}" for m in e["probe"]["manifests"]),
         "passed": all(m["status"] == "ok" for m in e["probe"]["manifests"])},
        {"criterion": "deliverables (a)-(f) as v3: L1 table, L2 arrays with flags/phases/masks, "
                      "domestic-excluded variant, Yunlin consistency, TPC comparison, cards",
         "threshold": "delivered",
         "measured": "census_L1/v3, preorigin/v3, full/v3 with PRODUCT_CARD.md; TPC comparison "
                     "descriptive (round 1, on v1 full; v3 full differs only through the "
                     "kept-set flips)", "passed": True},
    ]
    products = []
    for name, layer, uses, fwe in (
            ("census_L1", "L1", "model_design, prior, prior_width_input, design_time_evidence, "
                                "validator, after_the_fact_analysis (pre-origin uses through "
                                "2 exemptions)", "2026-07 (non_exempt 2022-12)"),
            ("preorigin", "L2", "forcing_frozen, model_design, prior, prior_width_input, "
                                "design_time_evidence, validator, after_the_fact_analysis "
                                "(pre-origin uses through 2 exemptions; no scoring_target, "
                                "no selection)", "2026-07 (non_exempt 2022-12)"),
            ("full", "L2", "forcing_recorded_arm (U4), statistical_layer_history, validator, "
                           "after_the_fact_analysis", "2026-07")):
        products.append({"path": f"{PE}/{name}/v3", "layer": layer,
                         "manifest": f"{PE}/{name}/v3/MANIFEST.json (+ PRODUCT_CARD.md, "
                                     "validation.json)", "allowed_uses": uses,
                         "fit_window_end": fwe})
    new = {
        "wp": "C0", "status": "passed",
        "summary": summary(e),
        "prereg": {"id": "C0.census_product.v3", "sha256": e["prereg_sha"],
                   "created": "2026-10-13T08:29:16.693365Z",
                   "superseded": ["C0.census_product.v2 (gate failed)",
                                  "C0.census_product.v1 (round 1)"]},
        "acceptance": acc, "products": products,
        "deviations_from_plan": deviations(e), "issues_for_downstream": issues(e),
        "code_paths": sorted(str(p) for p in lib.CODE_DIR.glob("*.py")),
        "repair_round": {
            "verifier_critical": "U3/U4 breach of preorigin/v1 (Dec-2022 of phase-B meters = "
                                 "half of a Dec-2022/Jan-2023 bill), fixed in v3",
            "evidence": {"v2": lib.rel(EV2), "v3": lib.rel(EV3)},
            "number_ids": QUOTED},
        "key_numbers": {
            "in_grid_GWh_per_yr_2012_2022_preorigin_v3": {
                t: r10["preorigin"][t]["in_grid_GWh_per_yr_2012_2022"] for t in C},
            "in_grid_GWh_per_yr_2012_2022_full_v3": {
                t: r10["full"][t]["in_grid_GWh_per_yr_2012_2022"] for t in C},
            "domestic_share_2012_2022_v3": {
                t: r10["preorigin"][t]["domestic_share_2012_2022"] for t in C},
            "dec2022_in_grid_kWh": {"v1": sum(r8[t]["dec2022_in_grid_v1_kWh"] for t in C),
                                    "v3": r10["preorigin"]["dec2022_in_grid_kWh"]},
            "v1_paired_share_of_dec2022": {t: r8[t]["v1_paired_share_of_dec2022"] for t in C},
            "fill_minus_realised_rel": {t: r8[t]["fill_minus_actual_rel"] for t in C},
            "r7": {"class_x_cell_coverage_90": r7["class_x_cell"]["coverage_90"],
                   "county_yl_ch_coverage_90": r7["county_in_grid"]["coverage_90_yunlin_changhua"],
                   "sd_label": r7["sd_label"]},
            "r3_yunlin_vs_model": {"max_rel_month_2012_01_2022_11":
                                   r3["max_rel_month_2012_01_2022_11"],
                                   "rel_2022_12": rel_dec},
            "residual_exemption": res,
            "kept_flips_vs_v1": {t: [e["R1"]["per_county"][t]["flips_in_meters"],
                                     e["R1"]["per_county"][t]["flips_out_meters"]] for t in C},
        },
        "evidence": lib.rel(EV),
        "claude_api_tokens": 0, "gpu": "none",
        "round_1": r1,
    }
    return new


def main() -> int:
    e = load()
    sec7 = section7(e)
    rows = ROWS + rlg.v2_rows()
    path = ledger.append_rows(lib.WP, rows)
    allrows = ledger.read_all()
    conf = ledger.conflicts(allrows)
    unl = ledger.unledgered(QUOTED + ["C0.v2.gates", "C0.v2.R4.sd_violations",
                                      "C0.v2.T5.violation", "C0.v2.R2.invariance"], allrows)
    ev = {"rows_offered": len(rows), "fragment": lib.rel(path),
          "fragment_rows": sum(r["wp"] == lib.WP for r in allrows),
          "rows_all_fragments": len(allrows), "conflicts_all_fragments": len(conf),
          "unledgered_quoted_ids": unl, "number_ids_quoted": len(QUOTED)}
    lib.save_json(ev, EV3 / "report_ledger.json")
    if conf or unl:
        raise SystemExit(f"ledger problems: conflicts {conf[:5]}, unledgered {unl[:5]}")
    sec7 = sec7.replace("@@LEDGER@@", (
        f"`ledger/rows/C0.csv` has {ev['fragment_rows']} rows. The repair round offered "
        f"{len(rows)} rows for the {len(QUOTED)} number ids quoted in this section plus the "
        "v2 outcomes; exact duplicates are skipped on a rerun. There are "
        f"{ev['conflicts_all_fragments']} conflicts over {ev['rows_all_fragments']:,} rows of "
        "all fragments, and every quoted id has rows. `ledger/data_roles.csv` was not "
        "reassembled (shared file)."))
    update_md(e, sec7)
    js = update_json(e)
    lib.save_json(js, JS)
    print(ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

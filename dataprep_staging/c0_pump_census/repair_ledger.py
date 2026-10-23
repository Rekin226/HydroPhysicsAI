"""WP C0 repair round: ledger rows (plan 4.1, U6) for every number the 'Repair round' section of
the C0 report quotes. One role per source per number; appended to ledger/rows/C0.csv.

``python -I repair_ledger.py v3`` writes the rows of the v3 checks (number ids C0.v3.*) plus the
rows of the v2 registered outcomes the report quotes (C0.v2.*).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import ledger  # noqa: E402

PE = "results/dataprep/pump_electricity"
CK2 = lib.EVIDENCE / "repair" / "checks"
CK3 = lib.EVIDENCE / "repair3" / "checks"
PKL = "results/twin_runs/stage3_datum_gate_cwa/inputs.local.pkl"
GRID = "results/twin_forward/datum_gate_innov.npz"


def l0(tag: str) -> list[str]:
    return [lib.rel(lib.PARQ / f"{tag}_{k}.parquet") for k in ("census", "kwh_long")]


L0_ALL = l0("yunlin") + l0("changhua") + l0("chiayi")


def rows_for(number_id: str, version: str, roles: dict[str, list[str]], note: str) -> list:
    return [{"number_id": number_id, "source": s, "role": role, "product_version": version,
             "note": note} for role, sources in roles.items() for s in sources]


def v2_rows() -> list:
    p2 = lib.load_json(lib.EVIDENCE / "repair" / "products_v2.json")
    PRE2, FULL2 = p2["preorigin"], p2["full"]
    e = {k: lib.load_json(CK2 / f"{k}.json") for k in ("R1", "R2", "R4", "T5", "summary")}
    R = rows_for("C0.v2.gates", PRE2, {"input": L0_ALL + [PRE2, FULL2, p2["census_L1"]]},
                 f"v2 registered gates: {json_gates(e['summary'])}")
    R += rows_for("C0.v2.R4.sd_violations", PRE2, {"input": [PRE2]},
                  f"v2 R4 FAIL: {e['R4']['preorigin']['violations']} violations, all in the sd "
                  f"recomputation (check used the pump class; meter class matches to 9e-13)")
    R += rows_for("C0.v2.T5.violation", PRE2, {"input": [PRE2, FULL2]},
                  f"v2 T5 FAIL: {e['T5']['violations']} class x cell entry at 2022-12, max "
                  f"{e['T5']['items']['cells_dec2022_without_fill']['max_abs']} kWh")
    R += rows_for("C0.v2.R2.invariance", PRE2, {"input": L0_ALL + [PRE2]},
                  f"v2 R2 pass {e['R2']['pass']}, violations {e['R2']['violations']}")
    return R


def json_gates(summ: dict) -> str:
    return "; ".join(f"{k.split('/')[-1]} {v['passed']}" for k, v in summ["gates"].items())


def main(version: str = "v3") -> int:
    CK = CK3
    prods = lib.load_json(lib.EVIDENCE / "repair3" / "products_v3.json")
    PRE, FULL, L1 = prods["preorigin"], prods["full"], prods["census_L1"]
    PRE1, L11 = f"{PE}/preorigin/v1", f"{PE}/census_L1/v1"
    e = {k: lib.load_json(CK / f"{k}.json") for k in ("R1", "R2", "R3", "R4", "R5", "R6",
                                                     "R7", "R8", "R9", "R10", "T5")}
    dl = lib.load_json(lib.EVIDENCE / "repair" / "design_look.json")
    R = []
    for t in lib.COUNTIES:
        r = e["R1"]["per_county"][t]
        R += rows_for(f"C0.v3.R1.flips.{t}", PRE, {"input": l0(t) + [PRE, L1],
                                                  "validator": [PRE1, L11]},
                      f"kept-meter flips v2 vs v1: +{r['flips_in_meters']} / "
                      f"-{r['flips_out_meters']}; pre part pass {r['preorigin']['pass']}, full "
                      f"part pass {r['full']['pass']}")
        R += rows_for(f"C0.v3.R1.filled_pumps.{t}", PRE, {"input": l0(t) + [PRE]},
                      f"{r['preorigin']['n_filled_pumps']} filled of "
                      f"{r['preorigin']['n_paired_pumps']} Dec-Jan-paired kept pumps")
        r2 = e["R2"]["per_county"][t]
        R += rows_for(f"C0.v3.R2.invariance.{t}", PRE, {"input": l0(t) + [PRE, L1],
                                                       "validator": [PRE1]},
                      f"perturbed L0 ({r2['perturbation']['pump_months_changed']} pump-months "
                      f"changed): {r2['violations']} violations; v1 builder control "
                      f"{r2['control_v1_builder']}")
        dc = dl["counties"][t]["R_det"]
        R += rows_for(f"C0.v3.design.flips.{t}", PRE, {"design_time_evidence": l0(t)},
                      f"design look (pre-origin counts): paired kept meters "
                      f"{dc['kept_meters_in_rule']}, flips +{dc['kept_flips_dropped_to_kept']} / "
                      f"-{dc['kept_flips_kept_to_dropped']}, no base "
                      f"{dc['kept_rule_meters_without_octnov_2022_base']}")
        r8 = e["R8"]["per_county"][t]
        R += rows_for(f"C0.v3.R8.fill_vs_actual.{t}", PRE, {"input": [PRE],
                                                           "validator": l0(t) + [PRE1]},
                      f"after the fact: fill - actual {r8['fill_minus_actual_rel']}, paired "
                      f"share of v1 Dec-2022 {r8['v1_paired_share_of_dec2022']}, rule "
                      f"sensitivity {r8['rule_sensitivity']:.4f}, precision "
                      f"{r8['rule_precision']:.4f}, untreated share "
                      f"{r8['untreated_share_of_dec2022_in_grid_meter_level']}")
        r9 = e["R9"]["per_county"][t]
        R += rows_for(f"C0.v3.R9.snapshot.{t}", PRE, {"input": l0(t) + [PRE]},
                      f"pumps installed after 2022-12: {r9['pumps_install_after_2022_12']} "
                      f"(in grid {r9['pumps_install_after_2022_12_in_grid']}); rel change "
                      f"2012-2022 {r9.get('rel_change_2012_2022')}")
        for v, P in (("preorigin", PRE), ("full", FULL)):
            r10 = e["R10"][v][t]
            R += rows_for(f"C0.v3.R10.{v}.energy.{t}", P, {"input": l0(t) + [GRID]},
                          f"in-grid {r10['in_grid_GWh_per_yr_2012_2022']:.4f} GWh/yr 2012-2022, "
                          f"domestic {r10['domestic_share_2012_2022']:.4f}")
    R += rows_for("C0.v3.R3.yunlin_vs_model", PRE, {"input": l0("yunlin") + [PRE],
                                                    "validator": [PKL]},
                  f"v2: max monthly rel 2012-01..2022-11 "
                  f"{e['R3']['v2']['max_rel_month_2012_01_2022_11']:.3g}, sum|d| "
                  f"{e['R3']['v2']['sum_abs_kwh_2012_01_2022_11']:.0f} kWh, 2022-12 rel "
                  f"{e['R3']['v2']['rel_month_2022_12']:.4f}")
    R += rows_for("C0.v3.R4.violations", PRE, {"input": L0_ALL + [PRE, FULL, L1]},
                  f"pre {e['R4']['preorigin']['violations']}, full "
                  f"{e['R4']['full']['violations']}, pass {e['R4']['pass']}")
    R += rows_for("C0.v3.T5.violations", PRE, {"input": [PRE, FULL]},
                  f"{e['T5']['violations']} violations")
    R += rows_for("C0.v3.R5.privacy", PRE, {"input": [PRE, FULL, L1]},
                  f"{len(e['R5']['violations'])} violations over {e['R5']['n_files']} files")
    R += rows_for("C0.v3.R6.counties", PRE, {"input": L0_ALL + [lib.rel(lib.PARQ /
                                                                         "pingtung_census.parquet"),
                                                                 GRID, PRE, L1]},
                  f"{e['R6']['counts']}")
    r7 = e["R7"]
    R += rows_for("C0.v3.R7.backtest", PRE, {"validator": L0_ALL, "input": [PRE, L1]},
                  f"LOYO 2012-2021: class x cell coverage {r7['class_x_cell']['coverage_90']:.4f} "
                  f"(n {r7['class_x_cell']['n']}), county YL+CH "
                  f"{r7['county_in_grid']['coverage_90_yunlin_changhua']:.3f}, county median "
                  f"|err| {r7['county_in_grid']['median_abs_rel_error']:.4f}, bias "
                  f"{r7['county_in_grid']['mean_rel_bias']:.4f}; sd {r7['sd_label']}")
    r10 = e["R10"]["preorigin"]
    R += rows_for("C0.v3.R10.dec2022", PRE, {"input": L0_ALL + [PRE]},
                  f"Dec-2022 in-grid {r10['dec2022_in_grid_kWh']:.0f} kWh, filled cells "
                  f"{r10['dec2022_filled_cells']}, sd upper {r10['dec2022_sd_fan_total_upper_kWh']:.0f}")
    R += rows_for("C0.v3.R10.missing_expected", PRE, {"input": L0_ALL + [PRE]},
                  f"missing_expected share 2012-2022 {r10['missing_expected_share_2012_2022']:.5f}")
    p2 = lib.load_json(lib.EVIDENCE / "repair" / "products_v2.json")
    for name in ("preorigin", "census_L1"):
        for P in (p2[name], prods[name]):
            R += rows_for(f"C0.v2.exemption.census_attributes.{name}", P,
                          {"input": [lib.rel(lib.PARQ / f"{t}_census.parquet")
                                     for t in lib.COUNTIES]},
                          "exemption (U3) named in the v2 and v3 manifests: census attribute "
                          "table of the 2026 delivery, data_end 2026-07, "
                          "non_exempt_fit_window_end 2022-12; sensitivity R9 of each version")
    R += v2_rows()
    path = ledger.append_rows(lib.WP, R)
    allrows = ledger.read_all()
    conf = ledger.conflicts(allrows)
    lib.save_json({"rows_appended": len(R), "fragment": lib.rel(path),
                   "conflicts_all_fragments": len(conf), "rows_all_fragments": len(allrows)},
                  lib.EVIDENCE / "repair3" / "ledger_check.json")
    print(len(R), "rows;", len(conf), "conflicts over", len(allrows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))

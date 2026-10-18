"""WP C0: ledger rows (plan 4.1, U6) for every number the C0 report quotes. One role per source
per number; sources are explicit files or product directories.

Usage: python -I ledger_c0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import ledger  # noqa: E402

PE = "results/dataprep/pump_electricity"
PRE, FULL, L1 = f"{PE}/preorigin/v1", f"{PE}/full/v1", f"{PE}/census_L1/v1"
P1CC = "results/dataprep/county_electricity/census_check/v2"
TPC = "results/dataprep/county_electricity/tpc_observed/v1"
AMP_C, AMP_K = "AMP_V2/data/tpc_pumps.parquet", "AMP_V2/data/pump_kwh_all.parquet"
PKL = "results/twin_runs/stage3_datum_gate_cwa/inputs.local.pkl"
GRID = "results/twin_forward/datum_gate_innov.npz"


def l0(tag: str) -> list[str]:
    return [lib.rel(lib.PARQ / f"{tag}_{k}.parquet") for k in ("census", "kwh_long")]


L0_ALL = l0("yunlin") + l0("changhua") + l0("chiayi")


def rows_for(number_id: str, version: str, roles: dict[str, list[str]], note: str = ""
             ) -> list[dict]:
    out = []
    for role, sources in roles.items():
        for s in sources:
            out.append({"number_id": number_id, "source": s, "role": role,
                        "product_version": version, "note": note})
    return out


def main() -> int:
    ev = lib.load_json
    ck = lib.EVIDENCE / "checks"
    t = {k: ev(ck / f"{k}.json") for k in ("T1a", "T1b", "T2a", "T2b", "T3", "T4", "T5", "T6",
                                          "T7", "T8", "T9")}
    dg = ev(lib.EVIDENCE / "diagnostics.json")
    R = []
    R += rows_for("C0.T1a.max_abs_delta", PRE, {"input": [AMP_C, AMP_K, GRID],
                                                "validator": [PKL]},
                  f"model path re-run vs cached E_by_class: max |delta| "
                  f"{max(v['max_abs_delta'] for v in t['T1a']['per_class'].values())} kWh")
    R += rows_for("C0.T1b.max_abs_delta", PRE, {"input": [AMP_C, AMP_K, GRID],
                                                "validator": [PKL]},
                  f"C0 pipeline on AMP_V2: {t['T1b']['max_abs_delta']:.3g} kWh")
    R += rows_for("C0.T2a.max_abs_delta", PRE, {"input": l0("yunlin") + [AMP_K, GRID],
                                                "validator": [PKL]},
                  f"new L0 with AMP-absent readings masked: {t['T2a']['max_abs_delta']:.3g} kWh")
    R += rows_for("C0.T2a.absent_readings", PRE, {"input": l0("yunlin") + [AMP_K]},
                  f"{t['T2a']['pump_months_absent_in_amp']} pump-months, "
                  f"{t['T2a']['absent_kwh_2012_2022']} kWh in 2012-2022, one single-pump meter")
    R += rows_for("C0.T2a.differ_2025_07", PRE, {"input": l0("yunlin") + [AMP_K]},
                  "35,789 pump-months differ, all in 2025-07 (AMP partial month)")
    R += rows_for("C0.T2b.max_rel_month", PRE, {"input": [PRE, AMP_K], "validator": [PKL]},
                  f"{t['T2b']['a_max_rel_month_total_delta']:.3g} ({t['T2b']['a_worst_month']})")
    R += rows_for("C0.T2b.sum_abs_delta", PRE, {"input": [PRE, AMP_K], "validator": [PKL]},
                  f"{t['T2b']['b_sum_abs_delta_kwh']:.1f} kWh vs limit "
                  f"{t['T2b']['b_limit_kwh']:.1f}")
    R += rows_for("C0.T3.max_abs_delta", PRE, {"input": [PRE, FULL], "validator": [P1CC]},
                  f"{t['T3']['variants']['full']['max_abs_delta_GWh']:.3g} GWh")
    R += rows_for("C0.T4.annual_coverage", FULL, {"input": [FULL], "validator": [TPC]},
                  "census/TPC 農林漁牧 annual coverage by county and scope (descriptive)")
    R += rows_for("C0.T4.month_of_year_coverage", FULL, {"input": [FULL], "validator": [TPC]})
    R += rows_for("C0.T4.changhua_over_yunlin", FULL, {"input": [FULL], "validator": [TPC]})
    R += rows_for("C0.T4.monthly_correlation", FULL, {"input": [FULL], "validator": [TPC]})
    R += rows_for("C0.T4.p1_c8_reproduction", FULL, {"input": [FULL], "validator": [P1CC]},
                  f"max rel diff {t['T4']['p1_c8_reproduction']['max_rel_diff']:.3g}")
    R += rows_for("C0.T5.max_abs_delta", PRE, {"input": [PRE, FULL]}, "0.0 kWh")
    R += rows_for("C0.T6.violations", FULL, {"input": [PRE, FULL, L1] + L0_ALL},
                  f"{t['T6']['violations']}")
    R += rows_for("C0.T7.counts", L1, {"input": [L1, PRE, lib.rel(lib.PARQ /
                                                                  "pingtung_census.parquet"),
                                                 GRID]})
    R += rows_for("C0.T8.violations", PRE, {"input": [PRE, FULL, L1] + [
        lib.rel(lib.PARQ / f"{c}_census.parquet") for c in ("yunlin", "changhua", "chiayi",
                                                            "pingtung")]}, "0")
    R += rows_for("C0.T9.max_abs_delta", PRE, {"input": l0("changhua") + l0("chiayi") + [
        PRE, GRID]}, "0.0 kWh (Changhua and Chiayi)")
    for v, d in (("preorigin", PRE), ("full", FULL)):
        R += rows_for(f"C0.stats.{v}.in_grid_energy_by_county_class", d,
                      {"input": [d]}, "GWh/yr 2012-2022 by county and class")
        R += rows_for(f"C0.stats.{v}.seasonal_cv", d, {"input": [d]})
        R += rows_for(f"C0.stats.{v}.domestic_share", d, {"input": [d]})
        R += rows_for(f"C0.stats.{v}.flag_counts", d, {"input": [d]})
        R += rows_for(f"C0.stats.{v}.terminated_meters", d, {"input": [d]})
        R += rows_for(f"C0.stats.{v}.billing_phase", d, {"input": [d]})
    R += rows_for("C0.stats.full.in_grid_energy_by_year", FULL, {"input": [FULL]})
    R += rows_for("C0.stats.full.evensplit_effect", FULL, {"input": [FULL]},
                  f"{dg['stats_full']['evensplit_vs_primary']}")
    R += rows_for("C0.coding_sensitivity", L1, {"input": L0_ALL},
                  "kept meters/energy 2012-2022 under the alternative pre-install coding")
    R += rows_for("C0.build.kept_pumps", L1, {"input": L0_ALL},
                  "Yunlin 114,392 / Changhua 115,766 / Chiayi 26,056 (316 in grid)")
    ledger.append_rows(lib.WP, R)
    rows = ledger.read_all()
    bad = ledger.conflicts(rows)
    mine = [r for r in rows if r["wp"] == lib.WP]
    print("C0 rows", len(mine), "conflicts (all fragments)", len(bad))
    lib.save_json({"c0_rows": len(mine), "conflicts_all_fragments": bad,
                   "rows_all_fragments": len(rows)}, lib.EVIDENCE / "ledger_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append the WP S2 round-2 ledger fragment (plan 4.1, U6): one row per source per number.

Usage: python -I ledger_rows.py <deep_benchmarks version dir> <deep_benchmarks_printed dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

VENDOR = [rel_to_repo(L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet") for k in L.VENDOR_KEYS]
RAW = [rel_to_repo(L.OPEN_DIR / f) for f in L.OPEN_FILES]
PRINTED_INT = rel_to_repo(L.PRINTED_CSV)
CY_L0 = "data_fetch_api/new_measurements/wra_cy_monthly_reports_007001"
PRE_ENV = [rel_to_repo(L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet")
           for k in ("TKJS_281_29", "TKJS_282_29", "NTUH_281_29", "NTUH_282_29", "SJES",
                     "STES", "WNES")]


def rows(dv: str, pv: str) -> list[dict]:
    out = []

    def add(num, src, role, ver, note=""):
        out.append({"number_id": num, "source": src, "role": role, "product_version": ver,
                    "note": note})

    feeds = VENDOR + RAW
    # declared tests A and B (design-time units)
    for num in ("S2.A_units_within_2mm", "S2.A_max_abs_mm", "S2.A_median_abs_mm",
                "S2.B_min_r", "S2.gate_passed"):
        add(num, PRINTED_INT, "validator", dv, "printed integrated-report values (verified)")
        for f in feeds:
            add(num, f, "input", dv)
        add(num, "results/dataprep/preregistration/S2r2.deep_benchmarks_core.json", "input", dv,
            "declared rules and thresholds")
    # fresh test C
    for num in ("S2.C1_units_within_2mm", "S2.C1_max_abs_mm", "S2.C1_min_r", "S2.C1_pass",
                "S2.C2_units_within_2mm", "S2.C2_max_abs_mm", "S2.C2_pass",
                "S2.C_secondary_month_end", "S2.C_secondary_as_delivered"):
        add(num, pv, "validator", dv, "printed 007001 table (verified transcription)")
        for f in feeds:
            add(num, f, "input", dv)
        add(num, "results/dataprep/preregistration/S2r2.c_fresh_cy007001.json", "input", dv,
            "registered test")
    add("S2.C_secondary_printed_cy_vs_integrated", pv, "input", dv)
    add("S2.C_secondary_printed_cy_vs_integrated", PRINTED_INT, "validator", dv)
    # step catalogue and envelope (pre-origin fit)
    for num in ("S2.envelope_Q_d", "S2.injected_step_recall", "S2.sigma_day"):
        for f in PRE_ENV:
            add(num, f, "input", dv, "vendor hourly feed <= 2022-12 only")
    for num in ("S2.raw_steps_catalogued", "S2.plan_steps_found", "S2.vendor_stitches",
                "S2.identity_shares"):
        for f in feeds:
            add(num, f, "input", dv)
    # transcription of the 007001 series
    for num in ("S2.cy007001_reports", "S2.cy007001_rows_verified",
                "S2.cy007001_tokens"):
        add(num, CY_L0, "input", pv, "58 PDFs and 4 listing pages (SOURCE.json)")
    add("S2.reports_newer_than_2026_01", "WRA portal listing GetDataReport/007003 "
        "(fetched 2026-10-12)", "input", "S2_evidence/portal_check_2026-10-12.json")
    # secondary versions of A and B (reported, not gating)
    for num in ("S2.A_secondary_flag012", "S2.A_secondary_vendor_as_delivered",
                "S2.A_secondary_raw_without_catalogue", "S2.A_secondary_on_grid",
                "S2.B_candidates"):
        add(num, PRINTED_INT, "validator", dv, "printed integrated-report values (verified)")
        for f in feeds:
            add(num, f, "input", dv)
    add("S2.cy007001_cross_report_consistency", pv, "input", pv,
        "printed value of a month in its report against the prior-year column a year later")
    # round 3 (resume): reproduction, spot check, after-the-fact diagnostics
    r3 = "results/dataprep/reports/S2_evidence/r3"
    prod = f"results/dataprep/{dv}"
    add("S2.r3_rebuild_identical", prod, "input", dv, "published tables")
    add("S2.r3_rebuild_identical", f"{r3}/repro_check.json", "validator", dv,
        "scratch rebuild compared table by table")
    add("S2.r3_pandas_us_union_probe", "synthetic pandas date_range (no data)", "input",
        "pandas 2.3.3", "probe in dataprep_staging/s2_deep_benchmarks/r3_repro_check.py")
    add("S2.r3_cy007001_spot_check", CY_L0, "validator", pv,
        "poppler layout text of 5 reports (35 rows, 136 printed numbers) read by the round-3 "
        "author")
    add("S2.r3_cy007001_spot_check", pv, "input", pv, "table1_rows.parquet")
    for num in ("S2.r3_agency_adjustments_vs_catalogue", "S2.r3_vendor_at_adjustments"):
        add(num, CY_L0, "validator", "S2_evidence/r3",
            "19 probe-adjustment records copied from the 007001 reports (pages cited)")
        add(num, prod, "input", "S2_evidence/r3", "steps.csv, vendor_stitches.csv, daily")
    for f in VENDOR:
        add("S2.r3_vendor_at_adjustments", f, "input", "S2_evidence/r3")
    add("S2.r3_c1_failure_pattern", pv, "validator", dv,
        "after-the-fact diagnostic of the scored test C1 (look logged)")
    add("S2.r3_c1_failure_pattern", prod, "input", dv, "test_c_units.parquet")
    add("S2.r3_a_failure_steps", PRINTED_INT, "validator", dv,
        "after-the-fact look at the three failing A units (look logged)")
    add("S2.r3_a_failure_steps", prod, "input", dv, "validation.json, steps.csv, stitches")
    add("S2.r3_guard_access_probe", prod, "input", dv, "loader guard outcomes per use")
    add("S2.r3_guard_access_probe", pv, "input", pv, "loader guard outcomes per use")
    add("S2.product_row_counts", prod, "input", dv, "daily, monthly, merged, steps tables")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dv = rel_to_repo(Path(argv[0])).replace("results/dataprep/", "")
    pv = rel_to_repo(Path(argv[1]))
    path = ledger.append_rows("S2", rows(dv, pv))
    print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

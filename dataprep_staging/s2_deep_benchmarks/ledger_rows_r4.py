"""Rewrite the WP S2 ledger fragment for round 4 (plan 4.1, U6): one row per source per number.

Round 4 corrects one role: the printed integrated-report values behind tests A and B (and the
round-4 sd coverage on the same units) are design-time units, so their role is
``design_time_evidence`` (the round-3 fragment said ``validator``). The round-3 fragment is
kept unchanged as ``ledger/superseded/S2.r3.csv``; the new fragment keeps the
pre-registration rows written by the A1 tool and adds the rows below.

Usage: python -I ledger_rows_r4.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import LEDGER_DIR, LEDGER_ROWS, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

VENDOR = [rel_to_repo(L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet") for k in L.VENDOR_KEYS]
RAW = [rel_to_repo(L.OPEN_DIR / f) for f in L.OPEN_FILES]
FEEDS = VENDOR + RAW
PRINTED_INT = rel_to_repo(L.PRINTED_CSV)
CY_L0 = "data_fetch_api/new_measurements/wra_cy_monthly_reports_007001"
PV1 = "results/dataprep/deep_benchmarks_printed/v1"
PV2 = "results/dataprep/deep_benchmarks_printed/v2"
DV1 = "results/dataprep/deep_benchmarks/v1"
DV2 = "results/dataprep/deep_benchmarks/v2"
PRE = "results/dataprep/preregistration"
EV3 = "results/dataprep/reports/S2_evidence/r3"
EV4 = "results/dataprep/reports/S2_evidence/r4"
PRE_ENV = [rel_to_repo(L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet")
           for k in ("TKJS_281_29", "TKJS_282_29", "NTUH_281_29", "NTUH_282_29", "SJES",
                     "STES", "WNES")]


def rows() -> list[dict]:
    out = []

    def add(num, src, role, ver, note=""):
        out.append({"number_id": num, "source": src, "role": role, "product_version": ver,
                    "note": note})

    def feeds(num, ver, note=""):
        for f in FEEDS:
            add(num, f, "input", ver, note)

    # plan tests A and B (design-time units), v1 and v2 (same full-run units and values)
    for num in ("S2.A_units_within_2mm", "S2.A_max_abs_mm", "S2.A_median_abs_mm",
                "S2.A_p95_abs_mm", "S2.A_failing_units", "S2.B_min_r", "S2.B_scope",
                "S2.gate_passed", "S2.A_secondary_flag012",
                "S2.A_secondary_vendor_as_delivered", "S2.A_secondary_raw_without_catalogue",
                "S2.A_secondary_on_grid", "S2.B_candidates"):
        add(num, PRINTED_INT, "design_time_evidence", "deep_benchmarks/v1;deep_benchmarks/v2",
            "printed integrated-report values (two-read verified); units compared with the "
            "feeds before any rule was fixed (seeded look)")
        feeds(num, "deep_benchmarks/v1;deep_benchmarks/v2")
        add(num, f"{PRE}/S2r2.deep_benchmarks_core.json", "input", "deep_benchmarks/v1",
            "declared rules and thresholds")
        add(num, f"{PRE}/S2r4.deep_benchmarks_v2.json", "input", "deep_benchmarks/v2",
            "rules carried unchanged")
    # registered test C, scored on v1
    for num in ("S2.C1_units_within_2mm", "S2.C1_max_abs_mm", "S2.C1_r_per_sensor",
                "S2.C1_pass", "S2.C1_by_year", "S2.C2_units_within_2mm", "S2.C2_max_abs_mm",
                "S2.C2_pass", "S2.C_secondary_month_end", "S2.C_secondary_as_delivered"):
        add(num, PV1, "validator", "deep_benchmarks/v1",
            "printed 007001 table (two-read verified); registered S2r2.c_fresh_cy007001")
        feeds(num, "deep_benchmarks/v1")
        add(num, f"{PRE}/S2r2.c_fresh_cy007001.json", "input", "deep_benchmarks/v1",
            "registered test")
    add("S2.C_secondary_printed_cy_vs_integrated", PV1, "input", "deep_benchmarks/v1")
    add("S2.C_secondary_printed_cy_vs_integrated", PRINTED_INT, "validator",
        "deep_benchmarks/v1")
    # round 4: test C's statistic on v2 (after the fact) and the convention diagnostics
    for num in ("S2.r4_C1_on_v2", "S2.r4_C1_on_v2_by_year", "S2.r4_C2_on_v2",
                "S2.r4_diag_2021", "S2.r4_diag_conventions_by_year"):
        add(num, PV1, "design_time_evidence", "deep_benchmarks/v2",
            "units already scored on v1 and used for the pre-origin NTUH labels; after the "
            "fact")
        feeds(num, "deep_benchmarks/v2")
    add("S2.r4_diag_2021", PV2, "design_time_evidence", "deep_benchmarks/v2",
        "primary and re-printed values")
    # round 4: pre-origin labels
    for num in ("S2.r4_preorigin_labels", "S2.r4_ntuh_pair_r", "S2.r4_tkjs283_preorigin_r"):
        add(num, PV2, "design_time_evidence", "S2_evidence/r4/preorigin_labels.json",
            "printed 007001 primary printings up to 2022-12 (time slice)")
        feeds(num, "S2_evidence/r4/preorigin_labels.json", "cut before 2023-01-01")
        add(num, f"{PRE}/S2r4.deep_benchmarks_v2.json", "input",
            "S2_evidence/r4/preorigin_labels.json", "declared label rule")
    # round 4: v2 construction checks
    for num in ("S2.r4_full_run_reproduces_v1", "S2.r4_preorigin_rows_vs_v1",
                "S2.r4_tkjs2_flags_changed"):
        add(num, DV1, "input", "deep_benchmarks/v2", "published v1 tables")
        feeds(num, "deep_benchmarks/v2")
        add(num, f"{EV4}/preorigin_labels.json", "input", "deep_benchmarks/v2")
    for num in ("S2.r4_sd_coverage_test_A", "S2.r4_sd_scale_after_the_fact"):
        add(num, PRINTED_INT, "design_time_evidence", "deep_benchmarks/v2",
            "the test-A units (seeded look)")
        feeds(num, "deep_benchmarks/v2")
    add("S2.r4_tkjs283_daily_change_r", rel_to_repo(
        L.VENDOR_DIR / "ls-wra-dbm-obs__TKJS_283_29.parquet"), "input", "deep_benchmarks/v2")
    for f in RAW:
        add("S2.r4_tkjs283_daily_change_r", f, "input", "deep_benchmarks/v2",
            "raw TKJS(3) daily means, full period (after the fact)")
    # round 4: printed v2
    for num in ("S2.r4_printed_v2_rows", "S2.r4_printed_v2_later_vintage",
                "S2.r4_printed_v2_reprint_pairs", "S2.r4_printed_v2_gate",
                "S2.r4_printed_rounding_coverage"):
        add(num, PV1, "input", "deep_benchmarks_printed/v2", "every printing of v1")
    add("S2.r4_printed_v2_gate", PRINTED_INT, "input", "deep_benchmarks_printed/v2",
        "two-read verification share of the integrated values")
    add("S2.r4_printed_v2_integrated_rows", PRINTED_INT, "input", "deep_benchmarks_printed/v2")
    # round 4: checks and searches
    add("S2.reports_newer_than_2026_01", "WRA portal listing GetDataReport/007003 (fetched "
        "2026-10-12T23:31 local)", "input", f"{EV4}/portal_check_2026-10-12T23.json")
    add("S2.reports_newer_than_2026_01", "WRA portal listing GetDataReport/007001 (fetched "
        "2026-10-12T23:3x local)", "input", f"{EV4}/listing_007001_2026-10-12T23.json")
    for num in ("S2.r4_guard_probe", "S2.r4_checker", "S2.r4_repro"):
        add(num, DV2, "input", "deep_benchmarks/v2")
        add(num, PV2, "input", "deep_benchmarks_printed/v2")
    add("S2.r4_checker", DV1, "input", "deep_benchmarks/v1")
    add("S2.r4_checker", PV1, "input", "deep_benchmarks_printed/v1")
    add("S2.r4_pandas_probe", "synthetic pandas date_range (no data)", "input", "pandas 2.3.3")
    # step catalogue and envelope (pre-origin fit; unchanged)
    for num in ("S2.envelope_Q_d", "S2.injected_step_recall", "S2.sigma_day"):
        for f in PRE_ENV:
            add(num, f, "input", "deep_benchmarks/v1;deep_benchmarks/v2",
                "vendor hourly feed <= 2022-12 only")
    for num in ("S2.raw_steps_catalogued", "S2.plan_steps_found", "S2.vendor_stitches",
                "S2.identity_shares", "S2.raw_feed_counts", "S2.product_row_counts"):
        feeds(num, "deep_benchmarks/v1;deep_benchmarks/v2")
    # transcription of the 007001 series (round 2)
    for num in ("S2.cy007001_reports", "S2.cy007001_rows_verified", "S2.cy007001_tokens",
                "S2.cy007001_values_verified"):
        add(num, CY_L0, "input", "deep_benchmarks_printed/v1",
            "58 PDFs and 4 listing pages (SOURCE.json)")
    add("S2.cy007001_cross_report_consistency", PV1, "input", "deep_benchmarks_printed/v1",
        "printed value of a month in its report against the prior-year column a year later")
    # round 3 evidence still reported
    add("S2.r3_cy007001_spot_check", CY_L0, "validator", "deep_benchmarks_printed/v1",
        "poppler layout text of 5 reports (35 rows, 136 printed numbers)")
    add("S2.r3_cy007001_spot_check", PV1, "input", "deep_benchmarks_printed/v1")
    for num in ("S2.r3_agency_adjustments_vs_catalogue", "S2.r3_vendor_at_adjustments"):
        add(num, CY_L0, "validator", "S2_evidence/r3",
            "19 probe-adjustment records copied from the 007001 reports (pages cited)")
        add(num, DV1, "input", "S2_evidence/r3", "steps.csv, vendor_stitches.csv, daily")
    for f in VENDOR:
        add("S2.r3_vendor_at_adjustments", f, "input", "S2_evidence/r3")
    add("S2.r3_a_failure_steps", PRINTED_INT, "design_time_evidence", "deep_benchmarks/v1",
        "after-the-fact look at the three failing A units (look logged)")
    add("S2.r3_a_failure_steps", DV1, "input", "deep_benchmarks/v1")
    add("S2.r3_pandas_us_union_probe", "synthetic pandas date_range (no data)", "input",
        "pandas 2.3.3", f"{EV3}/repro_check.json")
    return out


def main() -> int:
    frag = ledger.fragment_path("S2")
    sup = LEDGER_DIR / "superseded" / "S2.r3.csv"
    with open(frag, newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    keep = [r for r in old if r["number_id"].startswith("S2.prereg.")]
    if not sup.exists():
        sup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frag, sup)
    tmp = LEDGER_ROWS / ".S2.r4.tmp.csv"
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ledger.COLUMNS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in ledger.COLUMNS})
    new = rows()
    bad = ledger.conflicts(keep + [ledger._clean(r, "S2") for r in new])
    if bad:
        tmp.unlink()
        raise SystemExit(f"conflicts: {bad}")
    tmp.replace(frag)
    ledger.append_rows("S2", new)
    with open(frag, newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    print(f"{frag}: {n} rows ({len(keep)} pre-registration rows kept); round-3 fragment "
          f"kept as {sup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Look-log records of WP P2def (A1 tool ``harness.prereg3``), each written before the look it
describes runs (record_look cannot back-date). Electricity objects (census meters, pumps,
townships, TPC county sales) contain no registered well, leveling benchmark or model cell, so
their members are declared ``none``; head looks name their wells.

Usage (from code): looks_def.record(tag) before running the look.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import read_chained  # noqa: E402

NONE_ELEC = ("electricity object of WP P2def (TPC census pumps/meters/townships or TPC county "
             "sales); contains no registered well, leveling benchmark or model cell of the "
             "validation-unit registries")

LOOKS = {
    "P2def-d0-census-design": dict(
        units={
            "P2def:census_attributes_design": "census_L1 v3 attributes of kept pumps of every "
                                              "county: install-month validity, township "
                                              "coverage, install-year counts for years <= 2012 "
                                              "and >= 2016-07 only (2013-01..2016-06 counts are "
                                              "not computed: they hold the back-test truth and "
                                              "the 2013 surge)",
        },
        scope="census_L1_v3:2007-01..2026-07", purpose="design",
        note="Design-time facts for the P2d rules, before the P2def registration: kept pumps "
             "with a valid install month, townships, install-year counts outside the "
             "back-test cohort window 2014-01..2016-06 and outside 2013 (freeze check after "
             "2016-07; predictor window 2007-2012). No energy, no back-test quantity."),
    "P2def-d0-billing-completeness": dict(
        units={
            "P2def:billing_completeness_2016_2026": "kept installed meters of every county by "
                                                    "billing code and month 2016-01..2026-07: "
                                                    "counts of missing, zero and positive "
                                                    "readings and their kWh (C0 full v3)",
        },
        scope="c0_pump_electricity_v3:2016-01..2026-07", purpose="design",
        note="Design-time check asked by WP P1 (hypothesis H2: an incomplete census in the "
             "newest months) before the P2e registration: billing-group completeness of the "
             "census 2016-2026 per county and billing code. Shapes the P2e candidate set and "
             "the interpretation of P1's T10 error; no TPC comparison is computed here."),
    "P2def-d0-tpc-parse": dict(
        units={
            "P2def:tpc_d007012_farm_months": "TPC d007012 農林漁牧 rows of 雲林縣 and 彰化縣: "
                                             "months present (no values compared)",
        },
        scope="tpc_d007012:2016-01..2026-08", purpose="design",
        note="Parse check of the L0 TPC file before the P2e registration: months present per "
             "county for the farm sector, duplicates, gaps. No ratio or value comparison."),
}


def logged_tags(wp: str = D.WP) -> set:
    out = set()
    for lk in read_chained(prereg3.PREREG_DIR / "looks.jsonl"):
        if lk.get("wp") == wp and lk.get("note", "").startswith("["):
            out.add(lk["note"].split("]")[0] + "]")
    return out


def record(tag: str, units: dict | None = None, scope: str | None = None,
           purpose: str | None = None, note: str | None = None, members: dict | None = None):
    """Record look ``tag`` (from LOOKS unless given explicitly); a tag is logged once."""
    if f"[{tag}]" in logged_tags():
        print("already logged:", tag)
        return None
    lk = LOOKS.get(tag, {})
    units = units if units is not None else lk["units"]
    scope = scope or lk["scope"]
    purpose = purpose or lk["purpose"]
    note = note or lk["note"]
    mem = {}
    for u, desc in units.items():
        if members and u in members:
            m = dict(members[u])
            m.setdefault("description", desc)
        else:
            m = {"description": desc, "none": NONE_ELEC}
        mem[u] = m
    rec = prereg3.record_look(sorted(mem), scope, D.WP, purpose, members=mem,
                              note=f"[{tag}] {note}")
    print("logged", tag, rec.get("time"))
    return rec


if __name__ == "__main__":
    record(sys.argv[1])

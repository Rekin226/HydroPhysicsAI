"""Look records of the S1a repair round r3 (S1ar) in the A1 look log (harness.prereg3).

Every read of data that a registered or planned test uses is logged BEFORE the read, except
the availability listing of 2026-10-17T20:5xZ, which this module logs after the fact (it read
survey dates and counts only, no ring depth value).

    python -m s1_rings.r3_looks availability    # after the fact: survey-date listing
    python -m s1_rings.r3_looks diagnose        # before the r3 diagnosis of (e) and (a)

Units are named so that no record names a fresh unit of a later r3 test: the fresh test of
this round is declared on new synthetic units (see ``r3_prereg``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402

from s1_rings.seed_looks import site_units  # noqa: E402

WP = "S1a"
REGISTERED_UNITS = {
    "S1a:synth.R.mixed_groupcond.v1": None,   # declared with members in the v1 test
    "S1a:synth.R.mixed_harness.v1": None,
}


def log_availability() -> dict:
    u = "S1a:availability.ring_survey_dates_2026-10-18"
    members = {u: {"none": "survey dates and counts of the 37 L0 compaction files (no ring "
                           "depth value, no registered well, benchmark or cell)",
                   "description": "availability listing of ring survey dates, all years"}}
    return prereg3.record_look(
        [u], "rings_availability:2003-01..2025-05", WP, "availability", members=members,
        note=("S1a repair round r3 (S1ar), LOGGED AFTER THE READ (read at about "
              "2026-10-17T20:50Z, logged at the time of this record): one listing of the 37 "
              "files in data_fetch_api/new_measurements/compaction with, per file, the first and "
              "last survey date and the number of survey dates per calendar year. No depth value, "
              "ring id coverage or statistic was read or computed. Purpose: find which windows "
              "outside the spent 2016-01..2019-06 (e) starts have records (pre-2012 records exist "
              "at 10 of the 14 sites, 2003-2008 starts, 2 surveys in 2012)."))


def log_diagnose() -> dict:
    sites = site_units()
    units = sorted(list(sites) + list(REGISTERED_UNITS))
    return prereg3.record_look(
        units, "rings+piles:2014-05..2021-12", WP, "diagnose", members=sites,
        note=("S1a repair round r3 (S1ar), BEFORE any method change: diagnosis of the failed "
              "(e) S1a.e_synthetic_mixed.v1 on its spent units and data (2014-05..2021-12 ring "
              "surveys, pre-origin pile rows): (1) re-execution of the registered pipeline for "
              "the 42 scored windows (primary and stress) with its internals (line likelihood "
              "ratios, sd_deep, pools, members), reproduction check against "
              "synthetic_rows.parquet; (2) b_deep of the UNINJECTED surveys per window and "
              "site (the rate-continuity error field), its split into a window-common and a "
              "site part, and against the jackknife sd; (3) the depth profile and the timing of "
              "the 2016-2018 anomaly (deep-section length and survey-to-survey depth scale per "
              "site, 2014-05..2021-12), to tell a physical deep anomaly from an instrument step "
              "in the reference surveys; (4) calibration of each evidence line and of P(A); "
              "(5) pool sizes and the verifier's full re-execution without the near-threshold "
              "forcing rule. Diagnosis only: no rule, threshold or registered result changes. "
              "The pre-2012 surveys are NOT read by this look."))


def log_second_look_e() -> dict:
    """Before the r3 method runs on the spent (e) windows and seeds (second look)."""
    return prereg3.record_look(
        sorted(REGISTERED_UNITS), "rings+piles:2014-05..2021-12", WP, "second_look_score",
        prereg_id="S1a.e_synthetic_mixed.v1",
        note=("S1a repair round r3 (S1ar): the declared r3 method (S1a.rings_bridged_rules.v3) "
              "run on the SPENT units of S1a.e_synthetic_mixed.v1 (its 42 windows and registered "
              "seeds), AFTER the fresh re-test S1a.e_synthetic_fresh.v3 was scored. A declared "
              "SECOND LOOK: conditional, never gating, reported beside the registered rows."))


def log_second_look_a() -> dict:
    """Before the corrected check (a) is computed on the 14 real sites (second look)."""
    sites = site_units()
    return prereg3.record_look(
        sorted(sites), "rings+piles:2019-01..2025-05", WP, "second_look_score", members=sites,
        note=("S1a repair round r3 (S1ar), rings_bridged/v3 build: the registered check (a) and "
              "the corrected check (a) (trend-based bridged gap compaction, declared in "
              "S1a.rings_bridged_rules.v3 before this build) on the 14 real sites, which are "
              "design-time units; the corrected (a) is a declared SECOND LOOK (a pass is "
              "conditional). Also (b) and the residual offset under the v3 rules."))


def main(argv=None) -> None:
    which = (argv or sys.argv[1:] or [""])[0]
    if which == "availability":
        rec = log_availability()
    elif which == "diagnose":
        rec = log_diagnose()
    else:
        raise SystemExit("usage: r3_looks {availability|diagnose}")
    print(json.dumps({k: rec.get(k) for k in ("time", "wp", "purpose", "units", "data_scope")},
                     ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()

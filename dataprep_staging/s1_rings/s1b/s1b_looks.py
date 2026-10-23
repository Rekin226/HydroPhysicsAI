"""WP S1b: log this WP's looks that are not scoring looks (A1 look log, harness.prereg3).

    python -I dataprep_staging/s1_rings/s1b/s1b_looks.py design      (run once, first)
    python -I dataprep_staging/s1_rings/s1b/s1b_looks.py availability (before the synthetic run)
    python -I dataprep_staging/s1_rings/s1b/s1b_looks.py availability2 (before its re-run,
        resumed session: final constants, step-se term, family-level alternatives)

``design``: a ``design_time`` record of what this session read before any S1b statistic was
computed (the pilot's GNSS/InSAR ring-gap tables, S1a's after-the-fact GNSS-implied scales, the
product being validated, metadata). It names the S1a ring-site units (members none), so it
does not block the S1b registration; the registration and the report disclose it.

``availability``: an ``integrity_check`` record for the availability-preserving synthetic run
(dates, flags and valid-pixel masks of the real inputs are read; every displacement value is
replaced by a synthetic one before any computation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

NONE = "ring compaction site; contains no registered calibration well, leveling benchmark or cell"


def ring_site_units() -> dict:
    return {f"S1a:ring_site.{s}": {"none": NONE, "description": f"MLCW ring site {s}"}
            for s in K.SITES}


def design() -> dict:
    m = ring_site_units()
    m["S1b:design.read_tables_2026-10-14"] = {
        "none": "tables and reports read by the S1b session (no registered member)",
        "description": "pilot gnss_vs_ring*.csv / insar_vs_ring_gap.csv, S1a.md GNSS "
                       "sensitivity table, rings_bridged/v2 ring_corrections.csv"}
    note = (
        "S1b session (2026-10-14 local), logged before any S1b statistic was computed and before "
        "the S1b registration. Read: (1) the subsidence_gap_sources pilot's out/gnss_vs_ring.csv, "
        "out/gnss_vs_ring_stepfixed.csv (GNSS-implied scales 25-569 ppm at 7 sites, Honglun "
        "25 ppm) and out/insar_vs_ring_gap.csv (InSAR gap subsidence at 9 sites; corr 0.87 "
        "bridged, 0.10 raw per the plan) and the scripts gnss.py, gnss_stepfix.py, "
        "insar_pilot.py; (2) reports/S1a.md in full, including S1a's after-the-fact GNSS "
        "sensitivity on gnss_clean/full/v1 (GNSS-implied scales at 9 sites: Guangfu 165+-57, "
        "Xinsheng 204+-27, Xiutan 151+-13, Yuanchang 182+-8, Hunan 117+-34, Honglun -29+-29, "
        "Zhengmin -31+-21, Kecuo 73+-23, Tuku 90+-43 ppm; Kecuo's in-gap KTES step); (3) "
        "rings_bridged/v2 ring_corrections.csv (all columns, the product being validated) and "
        "README.md; (4) gnss_clean/full/v2 PRODUCT_CARD.md, stations.csv and steps.csv "
        "(metadata and the step catalogue; no height values); (5) insar_069A/v2 PRODUCT_CARD.md "
        "and network_segments.json (no displacement values); (6) ring-site and station "
        "coordinates (distances); (7) the S3, S4b and S1a summaries in the task brief and "
        "reports/A0.md, A1.md (harness).")
    return prereg3.record_look(sorted(m), "rings+piles+gnss+insar:2014-05..2025-12", K.WP,
                               "design_time", members=m, note=note)


def availability() -> dict:
    m = {"S1b:availability.synthetic_run_2026-10-14": {
        "none": "dates, flags and valid-pixel masks at the 14 ring sites and their GNSS stations; "
                "no displacement value enters any computation",
        "description": "availability-preserving synthetic run of the S1b pipeline (code test "
                       "and operating characteristics), before the S1b registration"}}
    note = ("s1b_synth.py: reads the real survey dates (both rings valid), GNSS daily flags and "
            "dates, and the InSAR valid-pixel/segment masks at the 14 ring sites, then replaces "
            "every compaction, height and displacement value by a synthetic one with a known "
            "group scale before any fit; reports pass rates of the registered rule under the "
            "correct and the wrong group (size and power). No real displacement value is "
            "computed or printed.")
    return prereg3.record_look(sorted(m), "rings+gnss+insar_availability:2020-01..2025-12",
                               K.WP, "integrity_check", members=m, note=note)


def availability2() -> dict:
    m = {"S1b:availability.synthetic_run2_2026-10-14": {
        "none": "dates, flags, valid-pixel masks and the GNSS step catalogue (dates, se) at "
                "the 14 ring sites and their GNSS stations; no displacement value enters any "
                "computation",
        "description": "re-run of the availability-preserving synthetic run with the final "
                       "constants, before the S1b registration (resumed session)"}}
    note = ("Resumed session (2026-10-14 local). The first synthetic run predates the last "
            "constant changes (MIN_SPAN_DAYS 270 -> 240, INSAR_END_MIN 2 -> 1; both from "
            "availability only, for Beichen). s1b_synth.py now also adds the se of every GNSS "
            "step the product removed inside the gap (read from steps.csv, already seen in the "
            "design look) to the bracket and to the synthetic series, and reports family-level "
            "pass rates under: the correct member, the registered P(A)-weighted mixture (truth "
            "= more probable group, and truth drawn with P(A)), a uniform +-25 % group-A scale "
            "error, the other member, and the raw value; plus the (d1) pass rate. Same "
            "availability-only reads as the first run; every value synthetic. Output: "
            "S1b_evidence/synthetic_operating_characteristics_v2.json.")
    return prereg3.record_look(sorted(m), "rings+gnss+insar_availability:2020-01..2025-12",
                               K.WP, "integrity_check", members=m, note=note)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else ""
    if what == "design":
        r = design()
    elif what == "availability":
        r = availability()
    elif what == "availability2":
        r = availability2()
    else:
        raise SystemExit("usage: s1b_looks.py design|availability|availability2")
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

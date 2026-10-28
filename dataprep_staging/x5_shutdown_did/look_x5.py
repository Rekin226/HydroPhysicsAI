"""Log WP X5 looks (harness.prereg3.record_look). Run under the shared registry lock:

    flock results/dataprep/preregistration/.x5.lock python -I look_x5.py --what availability
    flock results/dataprep/preregistration/.x5.lock python -I look_x5.py --what outcome

``availability``: the design-time look already taken before registration (survey dates,
benchmark and ring-site locations, township names, distances to the proxy wells; no elevation
or ring value). ``outcome``: logged immediately before run_x5.py reads the leveling and ring
values (design-time estimate under the declaration).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402
from harness import prereg3  # noqa: E402

LEV_UNIT = "X5:leveling_bm_ongrid_2008_2021"
RING_UNIT = "X5:rings_14_sites_2014_2021"


def members() -> dict:
    u = L.unit_table()
    ids = sorted(u.index[u.ongrid])
    sites = ["僑義國小", "元長國小", "光復國小", "內寮派駐站", "北辰國小", "嘉興國小", "土庫國中",
             "宏崙國小", "客厝國小", "拯民國小", "新生國小", "湖南國小", "溪州國小", "秀潭國小"]
    return {LEV_UNIT: {"description": "on-grid leveling benchmarks (within 1 km of an active "
                                      "twin cell), annual May surveys 2008-2021",
                       "ids": ids, "window": "2008-01..2021-12"},
            RING_UNIT: {"description": "the 14 compaction-ring sites of rings_bridged/v3, "
                                       "flag-0 primary series 2014-05..2021-12",
                        "ids": sites, "window": "2014-01..2021-12"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--what", choices=["availability", "outcome"], required=True)
    a = ap.parse_args(argv)
    m = members()
    if a.what == "availability":
        rec = prereg3.record_look(
            [LEV_UNIT, RING_UNIT], "leveling_dates+ring_sites:2008-01..2021-12", K.WP,
            "design_availability", members=m,
            note="before X5.shutdown_sdid.v1: survey dates and admissible intervals per "
                 "benchmark-year, benchmark/ring-site locations and townships, distances to "
                 "the 2022-inventory Yunlin Taiwan Water wells, unit counts per arm; ring "
                 "flag-0 month counts per site-year. No elevation, rate or compaction value "
                 "was read. Also a 2-panel/20-placebo synthetic dry run (engineering) and a "
                 "smoke run of run_x5.py on random numbers (no outcome read).")
    else:
        rec = prereg3.record_look(
            [LEV_UNIT, RING_UNIT], "leveling+rings:2008-01..2021-12", K.WP,
            "design_time_estimate", prereg_id=K.PREREG_ID,
            note="run_x5.py: SDID on leveling 2009-2021 and rings 2015-2021 (annual rates), "
                 "census kWh long difference; computed once under the declaration")
    print(rec.get("time"), rec.get("purpose"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

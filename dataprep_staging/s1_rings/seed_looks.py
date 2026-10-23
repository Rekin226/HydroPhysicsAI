"""Log the earlier looks at ring-gap results in the A1 look log (WP S1a, run once).

Two records, both written before any S1a rule was declared:

1. a back-dated ``design_time_seed`` for the subsidence_gap_sources pilot (2026-10-11
   16:43-17:02Z, file times of its outputs) and the plan revision's inline b_deep / group-rule
   runs (2026-10-12, time not recorded): real-gap scales, offsets, rate brackets, pile- GNSS-
   and InSAR-implied scales, residual offsets, the group-A-only synthetic pseudo-gaps g2017,
   g2018, g2019 and the interior-path test of masked years 2017-2021;
2. a ``design_time`` look by this session (S1a, 2026-10-13) that re-read those pilot tables,
   the printed pile monthly values 2020-10..2023-06 (deep_benchmarks_printed/v2) and vendor
   pile daily values around 2021-12 and 2022-12 (deep_benchmarks/v2) before writing any rule.

    python -m s1_rings.seed_looks          (from dataprep_staging/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402

from s1_rings.s1const import SITES  # noqa: E402

WP = "S1a"
NONE = "ring compaction site; contains no registered calibration well, leveling benchmark or cell"


def site_units() -> dict[str, dict]:
    return {f"S1a:ring_site.{s}": {"none": NONE, "description": f"MLCW ring site {s}"}
            for s in SITES}


def main() -> None:
    units = site_units()
    synth = {f"S1a:pilot_synth.groupA_only.{g}": {
        "none": "simulation draws on ring surveys (no registered member)",
        "description": f"pilot synthetic pseudo-gap {g} (group-A jumps only)"}
        for g in ("g2017", "g2018", "g2019")}
    path = {"S1a:pilot_interior_path.2017_2021": {
        "none": "masked calendar years of ring series (no registered member)",
        "description": "pilot interior-path test, masked years 2017-2021, 14 ring sites"}}
    m1 = {**units, **synth, **path}
    r1 = prereg3.record_look(
        sorted(m1), "rings+piles+gnss+insar:2014-05..2025-05", WP, "design_time_seed",
        members=m1, seeded_time="2026-10-11T17:02:48Z",
        note=("pilot subsidence_gap_sources (bridge_pilot.py, bridge_product.py, dbm_vs_ring.py, "
              "dbm_vs_ring_ntuh.py, gnss_stepfix.py, insar_pilot.py, jump_scan.py, gap_char.py; "
              "outputs 2026-10-11T16:43-17:02Z): real 2022-gap offsets, per-site deep-interval "
              "scales, sign-based groups, pooled scales, rate brackets (10/14 inside), pile-implied "
              "scales (153-203 ppm group A, -10 ppm at Zhengmin), GNSS-implied scales, InSAR gap "
              "correlation, residual offsets after correction (13 within 0.9 cm), the group-A-only "
              "synthetic windows g2017/g2018/g2019 (worst 1.35 cm) and the interior path of masked "
              "years 2017-2021 (pile 1.8 mm, climatology 5.3 mm). Also the plan revision's inline "
              "b_deep table and evidence-based group rule (2026-10-12, time not recorded, later "
              "than this seed). Seeded at the time of the pilot's last output."))
    m2 = dict(units)
    m2["S1a:pile_values.2020-10_2023-06"] = {
        "none": "deep-benchmark values (no registered well, benchmark or cell)",
        "description": "printed pile monthly values and vendor daily values read by S1a"}
    r2 = prereg3.record_look(
        sorted(m2), "rings+piles:2014-05..2025-05", WP, "design_time", members=m2,
        note=("S1a session 2026-10-13 re-read the pilot tables bridge_real_gap.csv, "
              "product/ring_corrections.csv, gap_sites.csv, bridge_synthetic.csv, jump_scan.csv, "
              "dbm_vs_ring.csv, dbm_vs_ring_ntuh.csv, gnss_vs_ring_stepfixed.csv, "
              "insar_vs_ring_gap.csv, the printed pile monthly values 2020-10..2023-06 "
              "(deep_benchmarks_printed/v2), printed cumulative 2021-03..2023-06 and the vendor "
              "pile daily values around 2021-12 and 2022-12 (deep_benchmarks/v2) before declaring "
              "any S1a rule; no S1a statistic was computed."))
    print(json.dumps({"seed": r1, "look": r2}, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

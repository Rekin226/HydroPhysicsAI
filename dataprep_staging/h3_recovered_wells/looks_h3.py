"""Look-log records of WP H3 (A1 ``prereg3``), written before the pre-registration.

Every look this WP made at head data of the candidate wells before ``H3.recovery.v1``:

* ``H3:inventory_2012-2022``: record extent and observed-month counts of all 344 stations
  (``s1_inventory.py``); no head value was compared with a prediction.
* ``H3:rh_probe``: availability of post-2018 records of the wells whose records stop in
  December 2018 (``s2_rh_probe.py``: sample counts per route and year, vendor record window).
* ``H3:switch_09030212``: while probing, the builder printed the December 2018 hourly-based
  daily means and the first ten once-daily values of January 2019 of well 09030212, and the
  standard deviation of its day-to-day changes in 2014-2018 (0.172 m, daily means) and
  2019-2022 (0.280 m, once-daily values). No snapshot-versus-monthly-mean error was computed.
* ``H3:layer_review_nantou6``: the layer review of the six no-code wells (``s3_layer_review``):
  the proposal rule's leave-one-site-out accuracy on coded wells, and the correlation of the
  six wells' monthly anomalies with their siblings and coded neighbours within 15 km.

Usage: python -I looks_h3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import pandas as pd  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from harness import prereg3  # noqa: E402


def main():
    inv = pd.read_csv(C.EVIDENCE / "inventory" / "inventory.csv", dtype={"sid": str})
    probe = pd.read_csv(C.EVIDENCE / "rh_probe" / "rh_probe_wells.csv", dtype={"sid": str})
    lr = json.loads((C.EVIDENCE / "layer_review" / "layer_review.json").read_text())
    near = lr["rule_loso_accuracy"]["within_15km_of_reviewed"]["wells"]
    recs = [
        ("H3:inventory_2012-2022", "heads:2012-01..2022-12",
         {"description": "record extent and observed-month counts (L1 v3 pre-origin, L0 cache "
                         "2012-2022) of every station; no prediction compared",
          "ids": sorted(inv.sid), "window": "2012-01..2022-12"},
         "s1_inventory.py: coverage definitions, exclusion reasons, candidate list"),
        ("H3:rh_probe", "heads:2018-01..2022-12",
         {"description": "post-2018 record availability of the wells whose L1 record ends in "
                         "2018-12 (sample counts per route and year; vendor record window)",
          "ids": sorted(probe.sid), "window": "2018-01..2022-12"},
         "s2_rh_probe.py: no prediction compared; 2023-25 sample counts are availability "
         "metadata only"),
        ("H3:switch_09030212", "heads:2018-12..2019-01",
         {"description": "09030212 daily means of December 2018 and once-daily values of "
                         "January 2019 printed; day-to-day change sd 2014-2018 and "
                         "2019-2022 printed", "ids": ["09030212"],
          "window": "2014-01..2022-12"},
         "builder's inspection during the R-h probe; no snapshot-vs-monthly error computed"),
        ("H3:layer_review_nantou6", "heads:2012-01..2022-12",
         {"description": "monthly anomaly correlations of the six no-code wells with their "
                         "siblings and coded wells within 15 km; leave-one-site-out accuracy "
                         "of the depth rule (metadata)",
          "ids": sorted(set(lr["reviewed"]) | set(near)), "window": "2012-01..2022-12"},
         "s3_layer_review.py"),
    ]
    for unit, scope, members, note in recs:
        r = prereg3.record_look([unit], scope, C.WP, "design_time", members={unit: members},
                                note=note)
        print(r["time"], unit)


if __name__ == "__main__":
    main()

"""Record the X5 looks in the shared registry (run under the X5 flock).

``prior``: the three design-time inspections made while designing X5 on 2026-10-25, before the
declaration (disclosed; none names the held-out unit):
  1. W1 janmay_validation.csv printed (observed Jan-May 2015-2021 both districts and W1's
     Jan-May sums of the same years);
  2. MOA 35644 Changhua/Yunlin office rows printed (crop seasons 2008-2015, annual 2016-2024);
  3. surface_water_v3.npz attributes printed (v3 crop totals per district 2012-2022, Jiji
     annual supply 2006-2025; a v3 model, not W1 v4).
``check``: the look on the held-out unit, logged immediately before the scorer reads it.

Usage: flock <lock> python -I look_x5.py prior|check
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402

from x5_changhua_water import x5const as K  # noqa: E402

NONE = "district or office aggregates of irrigation water; no wells, benchmarks or cells"
HELDOUT_UNIT = "X5:w1_yunlin_monthly_heldout"
HELDOUT_SCOPE = "w1_yunlin_monthly:2012-01..2014-12,2022-01..2022-12"


def prior() -> None:
    looks = [
        ("X5:w1_janmay_validation", "w1_janmay:2015-01..2021-12",
         "printed results/dataprep/surface_water_v4/full/v4/janmay_validation.csv (observed "
         "Jan-May supply 2015-2021 per district and W1's Jan-May sums) while designing X5"),
        ("X5:moa_office_totals", "moa_office:2004-01..2024-12",
         "printed the MOA 35644 Changhua and Yunlin office rows (crop seasons to 2015, annual "
         "after) while designing X5"),
        ("X5:v3_crop_totals", "surface_water_v3:2012-01..2022-12",
         "printed results/twin/surface_water_v3.npz attributes (v3 crop totals per district, "
         "Jiji annual supply) while designing X5; v3 is not the W1 v4 reference"),
    ]
    for unit, scope, note in looks:
        r = prereg3.record_look([unit], scope, K.WP, "design_time_inspection",
                                members={unit: {"none": NONE, "description": note}},
                                note=note)
        print("look", r["time"], unit)


def check() -> None:
    r = prereg3.record_look([HELDOUT_UNIT], HELDOUT_SCOPE, K.WP, "declared_check",
                            prereg_id=K.PREREG_ID, test_id=K.CHECK_ID,
                            note="agreement check scored by score_x5.py (W1 Yunlin monthly "
                                 "of the held-out years read now, once)")
    print("look", r["time"], HELDOUT_UNIT)


if __name__ == "__main__":
    {"prior": prior, "check": check}[sys.argv[1]]()

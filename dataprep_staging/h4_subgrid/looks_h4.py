"""WP H4: log H4's looks in the A1 look log (harness.prereg3.record_look).

Usage: python looks_h4.py <name>   with name one of LOOKS (each logged once per invocation).
Scoring looks are logged by prereg3.assert_can_score inside r1_registered.py, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harness import prereg3  # noqa: E402

PREREG_ID = "H4.subgrid_operator.v1"
UNIT = "H4:wells158_2020-2022"
LOOKS = {
    "d0": dict(
        units=["H4:d0_operator_no_heads"], data_scope="census_pumps:2007-01..2022-12",
        purpose="design_time",
        members={"H4:d0_operator_no_heads": {
            "none": "no head observation read: census pumps (coordinates, Chow-Lin kWh), the "
                    "engine/v2 hindcast and operator fields only",
            "description": "d0_design_check.py: pump counts, kernel shapes, unit-multiplier "
                           "correction sd, conversion check, placebo cost"}},
        note="before the pre-registration; reads no head data (results/dataprep/reports/"
             "H4_evidence/design_time/d0.json)"),
    "t0": dict(
        units=["H4:t0_synthetic_heads"], data_scope="synthetic_heads",
        purpose="design_time",
        members={"H4:t0_synthetic_heads": {
            "none": "synthetic observations built from the hindcast and the operator; no head "
                    "observation read",
            "description": "t0_smoke.py: pipeline smoke test, multiplier recovery (8, 4) -> "
                           "(7.95, 4.21)"}},
        note="before the pre-registration; reads no head data (results/dataprep/reports/"
             "H4_evidence/design_time/t0_smoke.json)"),
    "build": dict(
        units=[UNIT], data_scope="heads:2020-01..2022-12", purpose="product_build",
        prereg_id=PREREG_ID,
        note="b1_build_product.py packages the registered scores into subgrid_operator/v1; "
             "no new computation on head data"),
    "report": dict(
        units=[UNIT], data_scope="heads:2012-02..2022-12", purpose="report_facts",
        prereg_id=PREREG_ID,
        note="report_h4.py reads the registered evidence (scores.json, arrays.npz) to write "
             "the WP report: per-quartile, per-county and per-well descriptions after the "
             "registered score; no new fit"),
}


def main() -> None:
    name = sys.argv[1]
    kw = dict(LOOKS[name])
    rec = prereg3.record_look(kw.pop("units"), kw.pop("data_scope"), "H4", kw.pop("purpose"),
                              **kw)
    print(json.dumps({k: rec.get(k) for k in ("time", "purpose", "units", "data_scope")},
                     indent=1))


if __name__ == "__main__":
    main()

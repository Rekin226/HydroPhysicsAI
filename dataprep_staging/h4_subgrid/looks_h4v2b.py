"""WP H4 repair round: look definitions logged after the v2 product build.

looks_h4v2.py is hashed by subgrid_operator/v2 and stays unchanged; its 'build' and 'report'
entries named an undeclared unit (H4r:heldout_wells_2020_2022) and were refused by prereg3.
The 'build' look was logged with the registered unit H4r:heldout_wells_2012_2022 (2026-10-17
02:44:56Z, from a temporary edit that was reverted byte-exactly before the manifest check);
this module carries the corrected entries. Usage: python looks_h4v2b.py <name> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import prereg3  # noqa: E402

V2_PREREG = "H4.subgrid_operator.v2"
UNIT = "H4r:heldout_wells_2012_2022"
LOOKS = {
    "report": dict(
        units=[UNIT, "H4:wells158_2020-2022"], data_scope="heads:2012-02..2022-12",
        purpose="report_facts", prereg_id=V2_PREREG,
        note="report_h4v2.py reads the v2 registered evidence (scores.json, arrays.npz) to "
             "write the repair-round section of reports/H4.md; no new fit"),
    "guard_check": dict(
        units=[UNIT], data_scope="heads:2012-02..2022-12", purpose="product_check",
        prereg_id=V2_PREREG,
        note="guard probe and manifest re-hash of subgrid_operator/v2; no head data read"),
}


def main() -> None:
    for name in sys.argv[1:]:
        kw = dict(LOOKS[name])
        rec = prereg3.record_look(kw.pop("units"), kw.pop("data_scope"), "H4", kw.pop("purpose"),
                                  **kw)
        print(json.dumps({"name": name, "time": rec.get("time")}))


if __name__ == "__main__":
    main()

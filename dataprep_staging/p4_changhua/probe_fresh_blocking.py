"""WP P4.0 repair round, read-only probe of the A1 look log (harness.prereg3).

For a test registered now on each A1 fresh unit (validation_units/v3) with a 2012-22 head
scope: which looks would block it and which are disclosed as unverifiable. Writes nothing to
the log; optional argument: output JSON path.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
from harness import prereg3  # noqa: E402

reg = json.loads((REPO / "results/dataprep/validation_units/v3/units.json").read_text())
fresh = [u["id"] for u in reg["units"] if u.get("role") == "fresh"]
idx = prereg3.membership_index()
now = dt.datetime.now(dt.timezone.utc)
out = {"checked_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "n_fresh": len(fresh), "units": {}}
for f in fresh:
    t = {"test_id": f"probe.{f}", "fresh_units": [f], "data_scope": "heads:2012-01..2022-12"}
    blk = [b for b in prereg3.blocking_looks([t], before=None, idx=idx, test_time=now)]
    unv = [u for u in prereg3.unverifiable_looks([t], before=None, idx=idx)]
    out["units"][f] = {
        "blocking_all_wps": len(blk),
        "blocking_p40": sum(1 for b in blk if b["look"]["wp"] == "P40"),
        "blocking_wps": sorted({b["look"]["wp"] for b in blk}),
        "unverifiable_p40": sum(1 for u in unv if u["look"]["wp"] == "P40"),
        "unverifiable_all_wps": len(unv)}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")

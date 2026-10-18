"""WP S4a: apply the pre-registered fetch-budget rule to the two full-fan frames.

Rule (S4a.setup_acceptance.v1, ``fetch_rule``): after the primary frame is complete,
remaining = 3.0 GB minus every byte S4a transferred (``transfer_tally.json``, rerun
``tally.py`` first). Fetch 105D_06628_232108 only if its expected subset size x 1.10 fits in
the remaining; then 069A_06632_162010 under the same rule (with the remaining reduced by the
first frame's expected size x 1.10). A frame is fetched whole or not at all. Expected sizes come
from the header-only survey (``size_estimate.json``, ``expected_smart_clip_bytes``).

Writes ``results/dataprep/insar_tools/s4a_run/fetch_decision.json`` and prints the decision.
Usage (any env): python -I fetch_decision.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BUDGET_TOTAL_BYTES, RUN_DIR  # noqa: E402

ORDER = ("105D_06628_232108", "069A_06632_162010")
MARGIN = 1.10
PRIMARY = "069A_06694_131402"


def main() -> int:
    tally = json.loads((RUN_DIR / "transfer_tally.json").read_text())
    est = json.loads((RUN_DIR / "size_estimate.json").read_text())["frames"]
    val = RUN_DIR / f"validation_{PRIMARY}.json"
    v = json.loads(val.read_text()) if val.exists() else {}
    primary_complete = bool(v) and v.get("n_pairs_available_not_fetched") == 0
    remaining = BUDGET_TOTAL_BYTES - tally["total_bytes"]
    out = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rule": "S4a.setup_acceptance.v1 fetch_rule", "primary_complete": primary_complete,
           "tally_total_bytes": tally["total_bytes"], "remaining_bytes": remaining,
           "frames": {}}
    left = remaining
    for f in ORDER:
        need = est[f]["expected_smart_clip_bytes"] * MARGIN
        fetch = primary_complete and need <= left
        out["frames"][f] = {"n_pairs_available": est[f]["n_pairs_window"],
                            "expected_subset_bytes": est[f]["expected_smart_clip_bytes"],
                            "needed_with_margin_bytes": need, "remaining_before_bytes": left,
                            "fetch": bool(fetch),
                            "max_gb_for_run": round(left / 1e9 - 0.005, 3) if fetch else None}
        if fetch:
            left -= need
        before = out["frames"][f]["remaining_before_bytes"]
        print(f"{f}: need {need/1e9:.3f} GB (x{MARGIN}), remaining {before/1e9:.3f} GB -> "
              f"{'FETCH' if fetch else 'list only'}")
    (RUN_DIR / "fetch_decision.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

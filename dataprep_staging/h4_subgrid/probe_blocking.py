"""WP H4: read-only dry run of the A1 blocking rule (harness.prereg3) for candidate H4 tests.

Nothing is written: builds hypothetical test dicts (external units with members) and lists
the looks that would block or overlap them, so the pre-registration can name units that are
clean. Usage: python probe_blocking.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from harness import prereg3  # noqa: E402


def main() -> None:
    wells = pd.read_csv(DATAPREP / "validation_units/v3/wells.csv", dtype={"sid": str})
    ids = sorted(wells.sid)
    idx = prereg3.membership_index()
    now = _dt.datetime.now(_dt.timezone.utc)
    cands = {
        "ext_2020_22": {"test_id": "probe.ext_2020_22", "data_scope": "heads:2020-01..2022-12",
                        "fresh_units": ["H4probe:wells158_2020-2022"],
                        "members": {"ids": ids, "window": "2020-01..2022-12"}},
        "ext_2012_19": {"test_id": "probe.ext_2012_19", "data_scope": "heads:2012-02..2019-12",
                        "fresh_units": ["H4probe:wells158_2012-2019"],
                        "members": {"ids": ids, "window": "2012-02..2019-12"}},
        "registry_folds": {"test_id": "probe.reg_folds", "data_scope": "heads:2020-01..2022-12",
                           "fresh_units": ["fresh.folds.wells_all"], "members": None},
    }
    out = {}
    for k, t in cands.items():
        m = t.pop("members")
        if m is not None:
            mm = prereg3._declared_members({"description": k, **m})
            idx.members[idx.canon(t["fresh_units"][0])] = mm
        blk = prereg3.blocking_looks([t], None, idx=idx, test_time=now)
        unv = prereg3.unverifiable_looks([t], None, idx=idx)
        out[k] = {"n_block": len(blk), "block": [
            {"pairs": b["pairs"][:4], "look": b["look"]} for b in blk[:12]],
            "n_unverifiable": len(unv),
            "unverifiable": [{"units": u["units"][:4], "look": u["look"]} for u in unv[:8]]}
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

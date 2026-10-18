"""WP S5 repair round 2, fix round (S5r fix, 2026-10-18): look log entries.

    python -I s5r2f_looks.py fix_after      F.2 after-the-fact checks (exposure counts, T2/T2b
                                            calibration by distance, product regime); spent units
    python -I s5r2f_looks.py fix_report     reading fix-round evidence and v4 files for the report,
                                            the relabelled product v5 and the ledger; no new fit

Every entry is logged BEFORE its step reads data (A1, harness.prereg3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
from harness import prereg3  # noqa: E402

K_PREREG = "S5.repair_r2.v1"
LEV = ["s5.lev.t1_targets", "s5.lev.t2_blocks", "s5r2.lev.t1f_2018_2019"]

LOOKS = {
    "fix_after": dict(
        units=LEV, scope="leveling:2012-01..2021-12", purpose="after_the_fact",
        note=("S5 fix round, F.2 AFTER THE FACT (report only; nothing feeds a method, "
              "hyperparameter, threshold or gate): (A) exposure counts of the 2018-19 surveys in "
              "round-1 and round-2 T2 targets and in T1F targets; surveys never in a scored T1/T2 "
              "increment; (B) T2 and T2b raw coverage, rms z, median |z| and |z| > 3 share by "
              "distance to the nearest training benchmark; (C) distance from twin-grid cells to "
              "the nearest leveling benchmark. Declared in S5.repair.md F.2 (hashed) before this "
              "look.")),
    "fix_report": dict(
        units=LEV + ["s5.gnss.preorigin", "s5.insar.runP", "s5.cmp.preorigin",
                     "s5.post2022.families"],
        scope="subsidence_all:2012-01..2026-01", purpose="report_facts",
        note=("S5 fix round: reading fix-round evidence, round-2 evidence and the v4 product "
              "files to write the fix-round report section, S5.repair.json, the relabelled "
              "subsidence_fused/v5 (values copied from v4, labels corrected) and ledger rows; "
              "no new fit.")),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(LOOKS))
    a = ap.parse_args(argv)
    lk = LOOKS[a.what]
    rec = prereg3.record_look(lk["units"], lk["scope"], C.WP, lk["purpose"],
                              prereg_id=K_PREREG, note=lk["note"])
    print(json.dumps({k: rec[k] for k in ("time", "purpose", "units", "prereg_id",
                                          "scope_windows")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

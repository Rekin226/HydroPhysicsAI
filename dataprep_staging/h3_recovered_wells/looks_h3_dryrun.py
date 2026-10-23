"""Look-log record of WP H3's dry run of the scoring code (before ``H3.recovery.v1``).

To check that ``s5_runs.py`` and ``s6_score.py`` run before their bytes are hashed by the
registration, the builder ran them once with a stubbed ``assert_can_score``, an evidence
directory in the session scratchpad and H2's design realization r = 0 of family d. The run
computed (1) the bridge cells of r = 0 at the 16 test wells and (2) the ``H3.snapshot``
metric on well 09030212's own 2012-2018 months, and wrote them to scratch files that were
deleted unread. It printed only counts: 18 cells kept, scored windows at 10 of 16 wells,
09030212 with 75 own months in 26 quarters, 52 estimation wells, 33 once-daily months
passing the persistence screen; and that the rebuilt base matrix equals H2's archive. The
count of wells with a window led to 60 fresh realizations instead of 40.

Usage: python -I looks_h3_dryrun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from harness import prereg3  # noqa: E402


def main():
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    unit = "H3:dryrun_paths"
    members = {unit: {"description": "dry run of s5/s6 with a stubbed registration: family d "
                                     "r = 0 cells of the 16 test wells and the H3.snapshot "
                                     "metric computed into scratch files deleted unread; "
                                     "counts printed only",
                      "ids": sorted(set(reg["test_wells"])
                                    | set(reg["snapshot"]["estimation_wells"])),
                      "window": "2012-01..2022-12"}}
    r = prereg3.record_look([unit], "heads:2012-01..2022-12", C.WP, "design_time",
                            members=members,
                            note="code check before registration; no metric read")
    print(r["time"], unit)


if __name__ == "__main__":
    main()

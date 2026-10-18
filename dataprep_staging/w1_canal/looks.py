"""Record the looks at the W1 Jan-May benchmark in the A1 look log (plan 4.5).

The W1 pre-registration (results/dataprep/preregistration/W1_janmay_benchmark_2026-10-12.json,
written 2026-10-12T02:40Z by w1_canal/prereg.py, two minutes before the A1 tool
dataprep_staging/harness/prereg.py existed) is not in the A1 INDEX, so the A1 tool cannot
gate it with ``assert_can_score``. The looks are still logged in the hash-chained
``looks.jsonl`` so that no later test can treat these units as fresh:

    python -m w1_canal.looks seed      # the scout's look (design-time seed, back-dated)
    python -m w1_canal.looks dryrun    # the first W1 session's dry run (02:45Z), logged late
    python -m w1_canal.looks score     # a scoring run of the registered test (now)
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.prereg import record_look  # noqa: E402

from w1_canal.config import BENCH_YEARS, WP  # noqa: E402
from w1_canal.prereg import PREREG_ID  # noqa: E402

TEST_ID = "W1.janmay_benchmark"
UNITS = [f"W1.janmay.{d}.{y}" for d in ("chc", "yl") for y in BENCH_YEARS]
SCOPE = "wra_briefing_110_janmay:2015-01..2021-05"
#: mtime of the scout's validation_janmay.csv (first file scoring these units)
PILOT_LOOK_UTC = "2026-10-11T16:56:43.507315Z"
DRYRUN_NOTE = ("dry run of the registered test by the first W1 session, 2026-10-12T02:45:11Z "
               "(preorigin) and 02:46:26Z (full), outputs in that session's scratchpad "
               "(w1/stage1/); logged late by the continuing W1 session; the pre-registration "
               "(02:40:16Z) predates it")


def main(kind: str) -> dict:
    if kind == "seed":
        return record_look(UNITS, SCOPE, WP, "design_time_seed", seeded_time=PILOT_LOOK_UTC,
                           note="scout surface_water_monthly (pilot2.py, validation_janmay.csv) "
                                "scored supply-limited, calendar, climatological-fraction and "
                                "v3 Jan-May predictions on these units")
    if kind == "dryrun":
        return record_look(UNITS, SCOPE, WP, "score", test_id=TEST_ID, prereg_id=PREREG_ID,
                           note=DRYRUN_NOTE)
    if kind == "score":
        return record_look(UNITS, SCOPE, WP, "score", test_id=TEST_ID, prereg_id=PREREG_ID,
                           note="product build of the registered test (w1_canal.build)")
    raise SystemExit(f"unknown kind {kind!r}")


if __name__ == "__main__":
    print(main(sys.argv[1]))

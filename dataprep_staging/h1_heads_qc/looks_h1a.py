"""H1a look log (A1 harness): disclose earlier looks and gate each registered scoring step.

Usage:
  looks_h1a.py history            log the looks made before this resume (once; idempotent)
  looks_h1a.py score TEST_ID ...  prereg.assert_can_score for each test id (logs the look)

The harness stamps the logging time; the actual time of an earlier look is in its note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import prereg  # noqa: E402

PREREG_ID = "H1a.qc_acceptance.v1"
WP = "H1a"
MARK = "H1a-resume-history-2026-10-12"
PRE = "heads:2012-01..2022-12"

HISTORY = [
    {"units": ["H1a:inject_O_r0-9_paired", "H1a:inject_O_r0-9_step",
               "H1a:audit_artefact_events_hist", "H1a:o1_corrected_wells_2012_2022",
               "H1a:offset_episodes_2012_2022", "H1a:preorigin_rows"],
     "data_scope": PRE, "purpose": "smoke_test", "test_id": None, "prereg_id": None,
     "note": f"{MARK}: smoke runs BEFORE the pre-registration (about 2026-10-12T06:30Z-06:48Z) "
             "on 8 wells that are not real-event wells (07020111, 07020121, 07020131, 07010121, "
             "07040121, 07040141, 09180341, 09180331): injection realization r=0 (8 paired "
             "events, 8 recovered), audit/O1/pressure checks and the leak probe on those wells; "
             "disclosed in the pre-registration's interpretation; code bugs fixed only"},
    {"units": ["H1a:preorigin_rows"], "data_scope": PRE, "purpose": "score",
     "test_id": "H1a.leakprobe", "prereg_id": PREREG_ID,
     "note": f"{MARK}: leak probe computed by the first attempt at about 2026-10-12T07:01Z "
             "(after the pre-registration, without assert_can_score); result 0 differing rows "
             "of 904881; re-read on resume"},
    {"units": ["H1a:inject_O_r0-9_paired", "H1a:inject_O_r0-9_step"], "data_scope": PRE,
     "purpose": "score", "test_id": None, "prereg_id": PREREG_ID,
     "note": f"{MARK}: injection test computed by the first attempt at about "
             "2026-10-12T07:03Z (after the pre-registration); all 10 realizations ran but the "
             "summary crashed (pandas object dtype in .round) before any recall was written or "
             "printed; no outcome seen. Fixed in s5_inject.py (summary cast only, diff in "
             "H1a_evidence/postreg_fixes/s5_inject.diff) and rerun"},
]


def history() -> None:
    _, _, looks = prereg._paths(prereg.PREREG_DIR)
    done = any(MARK in (lk.get("note") or "") for lk in prereg.read_chained(looks))
    if done:
        print("history already logged")
        return
    for h in HISTORY:
        rec = prereg.record_look(h["units"], h["data_scope"], WP, h["purpose"],
                                 test_id=h["test_id"], prereg_id=h["prereg_id"], note=h["note"])
        print(json.dumps({k: rec[k] for k in ("time", "purpose", "test_id")}))


def score(test_ids: list[str]) -> None:
    for t in test_ids:
        r = prereg.assert_can_score(PREREG_ID, WP, test_id=t)
        print(json.dumps({"test_id": t, "prereg_sha256": r["sha256"],
                          "other_tests_prior_looks": r["other_tests_prior_looks"]}))


if __name__ == "__main__":
    if sys.argv[1] == "history":
        history()
    elif sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        raise SystemExit(__doc__)

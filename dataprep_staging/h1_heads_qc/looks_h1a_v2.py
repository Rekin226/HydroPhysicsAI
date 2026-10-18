"""H1a round 2 look log (A1 harness ``prereg2``): gate each registered v2 scoring step and log
the conformance and reported looks.

Usage:
  looks_h1a_v2.py score TEST_ID ...   prereg2.assert_can_score for each registered test
  looks_h1a_v2.py conformance         log the conformance checks (units seen in v1) and the
                                      reported post-origin analyses before s4_validate_v2 runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import prereg2  # noqa: E402

PREREG_ID = "H1a.qc_acceptance.v2"
WP = "H1a"

CONFORMANCE = [
    (["H1a:real_events_2024"], "heads:2024-05..2024-08",
     "C3 real-event conformance check of the v2 build (windows seen in v1)"),
    (["H1a:audit_artefact_events_hist"], "heads:2012-01..2022-12",
     "C4 audit-agreement conformance check (stage-1 jump catalogue unchanged from v1)"),
    (["H1a:o1_corrected_wells_2012_2022"], "heads:2012-01..2022-12",
     "C5 O1 coherence conformance check (barometric fits unchanged from v1)"),
    (["H1a:offset_episodes_2012_2022"], "heads:2012-01..2022-12",
     "C6 pressure-drop conformance check on the v2 flagged objects (holds by construction) "
     "and the year-shift chance-coincidence diagnostic"),
    (["H1a:post_origin_reported"], "heads:2023-01..2026-09",
     "reported only: post-origin analogues, frozen-challenge rescore, v1->v2 transitions"),
]


def score(test_ids: list[str]) -> None:
    for tid in test_ids:
        r = prereg2.assert_can_score(PREREG_ID, WP, test_id=tid)
        print(json.dumps({"test_id": tid, "sha256": r["sha256"],
                          "post_registration_looks": r["post_registration_looks"]}))


def conformance() -> None:
    for units, scope, note in CONFORMANCE:
        rec = prereg2.record_look(units, scope, WP, "conformance_check", prereg_id=PREREG_ID,
                                  note=note)
        print(json.dumps({k: rec[k] for k in ("time", "purpose", "units")}))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:])
    elif sys.argv[1] == "conformance":
        conformance()
    else:
        raise SystemExit(__doc__)

"""Scoring authorisation for the round-2 registered tests (A1 ``harness.prereg3``).

``authorize(test_ids)`` calls ``prereg3.assert_can_score`` for each test immediately before
its score is computed (it refuses a registration that postdates a look at the test data or
whose registered inputs changed, and logs the scoring look) and returns the records to be
stored with the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "P2abc.r2.c0_based.v1"


def authorize(test_ids, prereg_id: str = PREREG_ID) -> dict:
    out = {}
    for t in test_ids:
        r = prereg3.assert_can_score(prereg_id, L.WP, test_id=t)
        out[t] = {"prereg": prereg_id, "sha256": r["sha256"], "inputs_verified":
                  r["inputs_verified"], "post_registration_looks": r["post_registration_looks"],
                  "unverifiable_prior_looks": r["unverifiable_prior_looks"]}
    return out

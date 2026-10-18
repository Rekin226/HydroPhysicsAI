"""WP P2abc repair round (round 3, 2026-10-14): shared helpers.

The round-2 code in ``../v2`` is hashed in the manifests of yunlin_electricity/{preorigin,
full}/v1 and stays unchanged; this package imports it read-only (``p2lib`` as ``L``) and adds
the repair-round code. Products of this round are new versions (v2) of both products.

Repair items (independent verification of round 2, 2026-10-14):

* **P2c.** The registered zero check of round 2 scored a re-implementation of the pilot's
  s05(e) (``t4_zero_port.pilot_stats``), not the classifier that wrote ``zero_class.parquet``
  (``b2_zeros.classify``, which used different rules). This round ships a classifier whose
  run kinds are the pilot's rules (:mod:`zeros3`) and scores the shipped function itself.
* **P2b.** The full product left the plan's M0 July-2025 flag 9 unapplied and kept a deficit of
  phase-A July-August 2025 bills (missing bills recorded as zero) unflagged. This round
  applies the plan's rules to the full product (:mod:`mask2025`).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parent
V2 = V3.parent / "v2"
for _p in (V2,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import p2lib as L  # noqa: E402

WORK3 = L.PRODUCT_ROOT / "_work_p2abc_v3"
EVID3 = L.EVID / "repair3"
WP = L.WP


def code_sha(*paths) -> dict:
    return {L.rel(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def v3_code() -> dict:
    return code_sha(*sorted(V3.glob("*.py")))

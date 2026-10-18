"""WP P4m: record every P4m look in the A1 look log (harness.prereg3), with members.

Called by each P4m run script right after it computes (``record``); a run's tag makes the
call idempotent. Units outside the registries carry members:

* in-sample units read the calibration heads of the 158 wells over 2012-01..2022-12 (the
  cwa theta's own fit data); under prereg3's rule such a look is eligible to block later
  registrations on overlapping A1 fresh units, as P4.0's in-sample looks already do;
* numerical units compare model output with model output (``none``);
* E8 reads the hourly L0 records of the 158 wells 2012-01..2022-12; BE and E9 read H1a v3
  pre-origin rows of their stations;
* the recorded arm reads the 2023-01..2025-06 heads of the forcing experiment's
  observations (validator only, U4/U7).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PREREG_DIR  # noqa: E402
from harness import prereg3  # noqa: E402

WP = "P4m"
NUMERICAL_NONE = ("numerical check on the frozen twin's own forcing: model output against "
                  "model output or forcing totals against C0's own tables; no head, leveling "
                  "or other observation is read")


def have(tag: str) -> bool:
    looks = Path(PREREG_DIR) / "looks.jsonl"
    return any(lk.get("wp") == WP and tag in (lk.get("note") or "")
               for lk in prereg3.read_chained(looks))


def record(units: list[str], *, ids: list[str] | None, window: str | None, description: str,
           purpose: str, prereg_id: str, tag: str, note: str, none_reason: str | None = None,
           scope_name: str = "heads") -> bool:
    """Record one look on ``units`` (same members for each); False if ``tag`` is logged."""
    if have(tag):
        return False
    if none_reason is not None:
        members = {u: {"none": none_reason} for u in units}
        scope = f"numerical:{window or '2012-01..2022-12'}"
    else:
        m = {"ids": sorted(set(ids or [])), "description": description}
        if window:
            m["window"] = window
        members = {u: dict(m) for u in units}
        scope = f"{scope_name}:{window}"
    prereg3.record_look(units, scope, WP, purpose, members=members, prereg_id=prereg_id,
                        note=f"{tag} {note}")
    return True

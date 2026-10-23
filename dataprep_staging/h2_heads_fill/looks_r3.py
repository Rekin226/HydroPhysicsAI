"""Look-log records of WP H2 repair round 3 (A1 ``prereg3.record_look``; plan 4.5).

* ``verifier``  discloses the independent verifier's looks at H2 results (2026-10-14 04:22-04:31
  local time, 20:22-20:31Z on 2026-10-13: recomputation of every registered stratum from the
  saved run tables, a rerun of S r10, d r10 and E r30, a rebuild of the pre-origin v2 product,
  and two LOWO checks of the January 2012 initial condition against the twin's merged
  construction). Every unit involved was already scored; the record makes the history
  complete. Written before any repair-round-3 computation.

Usage: python -I looks_r3.py verifier
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402

VERIFIER_UNITS = [
    "H2:fresh_N_r10-19", "H2:fresh_S_r10-19", "H2:fresh_E_r10-19", "H2:fresh_H_r10-29",
    "H2:fresh_T_r10-29", "H2:fresh_d_r10-19", "H2:ic_lowo_2012-01", "H2:fresh2_N_r30-39",
    "H2:fresh2_S_r30-39", "H2:fresh2_E_r30-59", "H2:fresh2_H_r30-49", "H2:fresh2_T_r30-49",
    "H2:D_febmar_2013-2022"]


def verifier():
    rec = prereg3.record_look(
        VERIFIER_UNITS, "heads:2012-01..2022-12", C.WP, "verification_disclosure",
        note="Disclosure by H2 (repair round 3): the independent verifier of WP H2 looked at "
             "these already-scored units between 2026-10-13T20:22Z and 20:31Z (recompute.py: "
             "every registered stratum of H2.fill_acceptance.v1/v2 and H2.fresh.d from the "
             "saved run tables, including E strata with the pilot's E r0-1 well-gaps "
             "excluded; a rerun of S r10, d r10 and E r30 and a rebuild of "
             "heads/L2_filled_preorigin/v2; ic_check.py / ic_check2.py: LOWO of the twin's "
             "merged-proximal IDW and of the GP field on January 2012). No unit that repair "
             "round 3 registers as fresh (family d r >= 60; January 2013..2022 LOWO) was "
             "involved.")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units")}, indent=1))


def main(argv):
    if argv == ["verifier"]:
        verifier()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Look-log records of WP H2 (A1 harness ``prereg3.record_look``; plan 4.5).

Every H2 unit is declared with its members: the wells whose pre-origin heads the unit's
truth cells may come from (``ids``) and its month window. Design-time units are looked at
before the pre-registration; after the registration no look on an overlapping unit is made
until the registered scores are logged by ``prereg3.assert_can_score``.

Usage: python -I looks_h2.py design | history
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2  # noqa: E402

SCOPE = "heads:2012-01..2022-12"


def members(ids, desc, window="2012-01..2022-12"):
    return {"description": desc, "ids": sorted(ids), "window": window}


def design_units(h):
    ids = list(h.sids)
    d = {}
    for fam, rs in C.DESIGN_R.items():
        u = f"H2:design_{fam}_r{rs.start}-{rs.stop - 1}"
        d[u] = members(ids, f"H2 design-time realizations r={rs.start}..{rs.stop - 1} of "
                            f"family {fam} on the pre-origin L1 v3 matrix (truth cells: C20 "
                            f"months of these wells)")
    d["H2:D_febmar_2013-2022"] = members(
        ids, "family D: the 11 Feb - 19 Mar window removed in every year 2013-2022 "
             "(deterministic; scored by the pilot on 2013-2025)", "2013-01..2022-12")
    return d


def record_design(purpose="design_time", note=""):
    h = io_h2.load("pre", with_daily=False)
    units = design_units(h)
    rec = prereg3.record_look(sorted(units), SCOPE, C.WP, purpose, members=units,
                              note=note or "H2 design stage: router/baseline errors on "
                              "design-time realizations, gap-class sd factors, bridge sd "
                              "classes and per-class fall-backs chosen on these units only")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units")}, indent=1))


def main(argv):
    if argv and argv[0] == "design":
        record_design(*(argv[1:2] or []))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

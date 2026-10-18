"""WP S4b: the two looks logged after insar_069A/v1 was built (kept out of ``s4b_looks.py``,
which the product manifest hashes). Both were recorded on 2026-10-13 with these exact calls.

Usage (hydro env): python -I s4b_looks2.py <key>
  keys: dev_D_struct disclosure (logged after the fact): the structure-only tests of
                     s4b_evaluate on run D computed, without displaying them, GNSS, census and
                     Changhua-zone statistics on 2025 data
        explore_ramp exploratory, unregistered, after the fact: InSAR-minus-GNSS rate against
                     easting for the plane and offset + height references (run F)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
from harness import prereg3  # noqa: E402
from s4b_looks import all_cells  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["dev_D_struct", "explore_ramp"])
    a = ap.parse_args(argv)
    cells = all_cells()
    if a.key == "dev_D_struct":
        u = "s4b.insar105D.struct_tests.2025"
        rec = prereg3.record_look(
            [u], "insar_105D_gnss_census_coherence:2025-01..2026-01", C.WP, "development",
            members={u: {"description": "structure-only runs of s4b_evaluate (gnss, coherence, "
                                        "seasonal) with run D substituted for run P, about "
                                        "2026-10-13T19:27-19:31Z, before the registration",
                         "cells": cells, "window": "2025-01..2026-01"}},
            note="disclosure logged after the fact: the runs computed WRA LOSO (2025-02/03 "
                 "windows), Changhua-zone coherence fractions of 3 summer-2025 pairs and "
                 "amplitude-census correlations on 2025 data; only shapes, key names and "
                 "finite-ness were printed (one exception: the 3 per-pair fractions of active "
                 "cells with coherence >= 0.3 over the whole fan, 0.43-0.56, were printed by an "
                 "earlier output check). Post-origin and disjoint from every registered window; "
                 "nothing was changed in the registered design after them except the "
                 "undefined-competitor rule, which was added before registration")
    elif a.key == "explore_ramp":
        u = "s4b.wra_gnss.rates_2020_2025"
        rec = prereg3.record_look(
            [u], "insar_wra_gnss_rates:2020-01..2025-03", C.WP, "exploratory_after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "WRA GNSS stations vs run F, 2020-2025 rates, plane "
                                        "and offset + height references",
                         "ids": [f"wra_gnss:{s}" for s in sorted(
                             __import__("s4b_lib").wra_files())],
                         "window": "2020-01..2025-03"}},
            note="unregistered exploratory diagnostic of the reference ramp after the gate "
                 "failed; never a gate or a design choice for the registered product")
    else:
        u = "s4b.wra_gnss.rates_2020_2025"
        rec = prereg3.record_look(
            [u], "insar_wra_gnss_rates:2020-01..2025-03", C.WP, "exploratory_after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "WRA GNSS stations vs run F, 2020-2025 rates, plane "
                                        "and offset + height references",
                         "ids": [f"wra_gnss:{s}" for s in sorted(
                             __import__("s4b_lib").wra_files())],
                         "window": "2020-01..2025-03"}},
            note="unregistered exploratory diagnostic of the reference ramp after the gate "
                 "failed; never a gate or a design choice for the registered product")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units", "data_scope")
                      if k in rec}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

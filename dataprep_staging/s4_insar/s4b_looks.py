"""WP S4b: record looks at InSAR data in the A1 look log (``harness.prereg3.record_look``).

Each look is logged BEFORE the script that reads the values runs. Units outside the registries
are declared with members (prereg3): InSAR-derived statistics at twin cells name the active
cells they touch and the epoch window; GNSS-station comparisons name the stations.

Usage (hydro env): python -I s4b_looks.py <key>
  keys: dev_D        development of the post-processing on run D (descending, 2025-01..2026-01)
        dev_F_post   development checks on run F restricted to 2023-01..2025-12 (post-origin)
        after_F_pre  after-the-fact use of run F at pre-origin epochs (report-only rows,
                     after the registered tests were scored on run P)
        after_pilot  the pilot's post-origin GNSS window (2021-12..2023-02), report only
        decomp_2025  asc/desc decomposition over 2025 (after the fact)
        after_ngl    NGL on-fan stations vs InSAR (runs P and F), pre-origin window, report only
        after_C1_all coherence of all pre-origin pairs at Changhua zones, report only
        extract_C1   extraction of the summer-pair cell coherence for registered test C1
                     (logged before s4b_coherence.py reads the pre-origin cc files)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import s4b_config as C  # noqa: E402
from harness import prereg3  # noqa: E402


def all_cells() -> list[int]:
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    return list(range(int(np.asarray(z["mask"]).sum())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["dev_D", "dev_F_post", "after_F_pre", "after_pilot",
                                    "decomp_2025", "extract_C1", "after_ngl",
                                    "after_C1_all"])
    a = ap.parse_args(argv)
    cells = all_cells()
    if a.key == "dev_D":
        u = "s4b.insar105D.cells.2025"
        rec = prereg3.record_look(
            [u], "insar_105D_cum_coherence:2025-01..2026-01", C.WP, "development",
            members={u: {"description": "InSAR descending frame 105D_06628_232108 (run D) at "
                                        "the twin's active cells, epochs 2025-01..2026-01",
                         "cells": cells, "window": "2025-01..2026-01"}},
            note="code development of s4b_lib referencing, cell aggregation, coherence and "
                 "harmonic routines on run D (post-origin; disjoint from every registered "
                 "pre-origin test window); no GNSS, ring, census or Changhua-zone statistic")
    elif a.key == "dev_F_post":
        u = "s4b.insar069A.cells.2023_2025"
        rec = prereg3.record_look(
            [u], "insar_069A_cum:2023-01..2025-12", C.WP, "development",
            members={u: {"description": "InSAR ascending frame 069A (run F) at the active cells, "
                                        "epochs 2023-01..2025-12", "cells": cells,
                         "window": "2023-01..2025-12"}},
            note="checks of the run F product arrays at post-origin epochs only")
    elif a.key == "after_F_pre":
        u = "s4b.insar069A.runF.cells.2020_2022"
        rec = prereg3.record_look(
            [u], "insar_069A_runF_cum:2020-01..2022-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "run F (full network) values at pre-origin epochs, "
                                        "read after the registered tests were scored on run P",
                         "cells": cells, "window": "2020-01..2022-12"}},
            note="report-only comparison of the full-network product with the gate run")
    elif a.key == "after_pilot":
        u = "s4b.wra_gnss.pilot_window"
        rec = prereg3.record_look(
            [u], "insar_wra_gnss_change:2021-11..2023-03", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "WRA GNSS stations vs InSAR (run F) over the pilot's "
                                        "post-origin window 2021-12-05 -> 2023-02-20",
                         "none": "GNSS stations are not registry units; report only"}},
            note="report-only comparison with the pilot's 8.2 mm (post-origin window)")
    elif a.key == "after_ngl":
        u = "s4b.ngl_onfan.preorigin"
        rec = prereg3.record_look(
            [u], "insar_ngl_gnss_change:2021-08..2022-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "NGL on-fan stations DNAN, FUNY, HUWE, JSAN, PUS2 vs "
                                        "InSAR (runs P and F) with the WRA tie",
                         "ids": [f"ngl_gnss:{s}" for s in ("DNAN", "FUNY", "HUWE", "JSAN",
                                                          "PUS2")],
                         "window": "2021-08..2022-12"}},
            note="report-only row registered in S4b.insar_acceptance.v1")
    elif a.key == "after_C1_all":
        units = ["s4b.coh.Z1", "s4b.coh.Z2", "s4b.coh.Z3"]
        rec = prereg3.record_look(
            units, "insar_069A_coherence:2020-01..2022-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            note="report-only C1 row: every pre-origin pair, after C1 was scored")
    elif a.key == "extract_C1":
        units = ["s4b.coh.Z1", "s4b.coh.Z2", "s4b.coh.Z3"]
        rec = prereg3.record_look(
            units, "insar_069A_coherence:2020-06..2022-09", C.WP, "extraction",
            prereg_id=C.PREREG_ID, test_id="S4b.C1_changhua_summer_coherence_preorigin",
            note="s4b_coherence.py --rule summer_preorigin: per-cell median coherence of the "
                 "registered summer pairs, written to s4b_run/coh_P_summer_preorigin.npz; "
                 "scored by s4b_evaluate.py coherence")
    else:
        u = "s4b.insar_ascdesc.cells.2025"
        rec = prereg3.record_look(
            [u], "insar_069A_105D_rates:2025-01..2025-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "2025 LOS rates of runs F and D at the active cells",
                         "cells": cells, "window": "2025-01..2025-12"}},
            note="asc/desc decomposition, after-the-fact analysis")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units", "data_scope")
                      if k in rec}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

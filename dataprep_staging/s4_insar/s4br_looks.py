"""WP S4b repair round 2 (S4br, 2026-10-18): looks recorded in the A1 look log
(``harness.prereg3``) BEFORE the script that reads the data runs.

Usage (hydro env): python -I s4br_looks.py <key>

  diag_wra      s4br_diag.py: error budget of the registered G1 on the round-1 run P at the
                18 spent WRA stations (2021-08..2022-12): residual series, gap / within-segment
                split, epoch noise, InSAR-only and GNSS-only differential noise, station-pixel
                loop-closure quality, the reference-plane pattern, the S3 gnss_clean full/v4
                step catalogue and step-corrected series (after_the_fact_analysis), the 2025
                east rate at the station cells (insar_069A/v2, after the fact) and the 3x3 vs
                7x7 pixel-window change.
  diag_insar    s4br_diag.py: InSAR-internal diagnostics of runs P and F of round 1 (loop
                closure with and without the multi-prime bias, the interferograms spanning
                each network gap and why they were removed, the stable-strip geometry) and
                the pre-origin coherence statistics of the Changhua zones (C1 cells).
  diag_y        s4br_diag.py yconf: the annual cycle the round-1 stable-strip plane injects
                into the Yunlin cells, against the thickness proxy and the registered run-P
                amplitude (insar_069A/v2, spent Y units).

The NGL GNSS family is NOT read by any diagnostic: it is kept for the fresh re-test.
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

WP = "S4b"
G1_STATIONS = ["ANES", "CHSG", "FRES", "GFES", "HLES", "HNES", "JJES", "JMES", "KTES", "LYES",
               "NGES", "NTUH", "SLES", "SNES", "SSES", "STES", "YCES", "YWJS"]


def zone_cells() -> list[int]:
    u = json.loads(C.UNITS_V3.read_text())
    by = {x["id"]: x for x in u["units"]}
    return sorted(set().union(*[set(by[k]["cells"]) for k in C.CH_ZONES]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["diag_wra", "diag_insar", "diag_y"])
    a = ap.parse_args(argv)
    st_ids = [f"wra_gnss:{s}" for s in G1_STATIONS]
    if a.key == "diag_wra":
        us = ["s4br.diag.wra_gnss.g1_budget", "s4br.diag.wra_gnss.s3v4_steps"]
        rec = prereg3.record_look(
            us, "insar_wra_gnss_g1_budget:2021-08..2022-12", WP, "diagnose",
            prereg_id=C.PREREG_ID,
            members={
                us[0]: {"description": "round-1 run P vertical-equivalent at the 18 G1 WRA "
                                       "stations and their WRA L0 heights: registered-tie "
                                       "residual series, gap and within-segment split, epoch "
                                       "noise, InSAR-only and GNSS-only differential noise, "
                                       "loop-closure quality, reference-plane pattern, 2025 "
                                       "east rate at the station cells, 3x3 vs 7x7 window",
                        "ids": st_ids, "window": "2021-08..2022-12"},
                us[1]: {"description": "S3 gnss_clean full/v4 (after_the_fact_analysis): steps "
                                       "at the 18 G1 stations inside the G1 span and G1 "
                                       "recomputed with the step-corrected WRA heights",
                        "ids": st_ids, "window": "2021-08..2022-12"}},
            note="S4br diagnosis (repair round 2): after-the-fact error budget of the failed "
                 "G1 on spent units; never a gate, threshold or design choice. The NGL family "
                 "is not read.")
    elif a.key == "diag_y":
        from s4b_r2_looks import yunlin_cells
        u = "s4br.diag.yconf.yunlin"
        rec = prereg3.record_look(
            [u], "insar_069A_seasonal_amplitude:2020-01..2022-12", WP, "diagnose",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "annual amplitude the round-1 stable-strip east "
                                        "plane term injects at the Yunlin cells, against the "
                                        "west-of-pinch-out proxy and the registered run-P "
                                        "amplitude (insar_069A/v2)",
                         "cells": yunlin_cells(), "window": "2020-01..2022-12"}},
            note="S4br diagnosis of the Y failure (spent units, after the fact); no census "
                 "value is read")
    else:
        us = ["s4br.diag.insar.network_loops", "s4br.diag.coh.changhua"]
        rec = prereg3.record_look(
            us, "insar_licsbas_internal_and_coherence:2020-01..2025-12", WP, "diagnose",
            members={
                us[0]: {"description": "InSAR-internal: loop-closure phase of runs P and F "
                                       "(with and without the multi-prime bias), the "
                                       "interferograms spanning each network gap and the step "
                                       "that removed them, stable-strip geometry; no GNSS, "
                                       "ring, pile, leveling or census value",
                        "cells": all_cells(), "window": "2020-01..2025-12"},
                us[1]: {"description": "pre-origin LiCSAR coherence at the Changhua zone cells "
                                       "(the C1 cells): per pair temporal baseline, month and "
                                       "year; summer vs other pairs",
                        "cells": zone_cells(), "window": "2020-01..2022-12"}},
            note="S4br diagnosis (repair round 2): InSAR-internal structure and the C1 "
                 "coherence statistics behind the Z2 failure; no validator value")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units", "data_scope")
                      if k in rec}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WP S4b repair round (2026-10-14): looks recorded in the A1 look log (``harness.prereg3``).

Two are late disclosures of looks that happened before this round and were never logged; two
are logged BEFORE the repair scripts that read the data run.

Usage (hydro env): python -I s4b_r2_looks.py <key>
  keys: disclose_tie_offsets  the v1 product build (2026-10-13, about 19:47-19:49Z) computed
                              gnss_tie_offsets.csv: per run-F epoch 2020-01..2025-12, the mean
                              over the 18 G1 WRA stations of (InSAR - GNSS). Never logged.
        disclose_verifier     the independent verification of 2026-10-14 recomputed G1, C1 and
                              Y1, compared runs P and F at active-cell pixels over
                              2021-10..2022-12 (InSAR only), fitted harmonics to the run-P
                              stable-area reference coefficients and correlated the amplitude
                              and census density with easting. Logged here on its behalf.
        repair_gap_meta       s4b_r2_gaps.py: LiCSBAS network-gap structure of runs P, F, D
                              (which interferograms are valid per pixel; no displacement,
                              coherence, GNSS or census value)
        repair_diag           s4b_r2_diag.py: after-the-fact diagnostics of the G1 failure
                              (P vs F per interval class; G1 error split into within-segment
                              and cross-gap increments), the reference-coefficient harmonics and
                              drifts, and the easting confound of test Y
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

G1_STATIONS = ["ANES", "CHSG", "FRES", "GFES", "HLES", "HNES", "JJES", "JMES", "KTES", "LYES",
               "NGES", "NTUH", "SLES", "SNES", "SSES", "STES", "YCES", "YWJS"]


def yunlin_cells() -> list[int]:
    u = json.loads(C.UNITS_V3.read_text())
    by = {x["id"]: x for x in u["units"]}
    return sorted(set().union(*[set(by[k]["cells"]) for k in C.YUNLIN_UNITS]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=["disclose_tie_offsets", "disclose_verifier",
                                    "repair_gap_meta", "repair_diag"])
    a = ap.parse_args(argv)
    cells = all_cells()
    st_ids = [f"wra_gnss:{s}" for s in G1_STATIONS]
    if a.key == "disclose_tie_offsets":
        u = "s4b.wra_gnss.tie_offsets_2020_2025"
        rec = prereg3.record_look(
            [u], "insar_wra_gnss_tie_offsets:2020-01..2025-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={u: {"description": "per-epoch mean over the 18 G1 WRA stations of (run-F "
                                        "vertical-equivalent InSAR - WRA GNSS height), every "
                                        "run-F epoch 2020-01-12..2025-12-11; written to "
                                        "insar_069A/v1/gnss_tie_offsets.csv",
                         "ids": st_ids, "window": "2020-01..2025-12"}},
            note="LATE DISCLOSURE (S4b repair round, 2026-10-14): computed by "
                 "s4b_build_product.py during the v1 build on 2026-10-13 at about "
                 "19:47-19:49Z and not logged at the time. Any later registration on these "
                 "WRA stations in 2020-01..2025-12 must treat them as looked at.")
    elif a.key == "disclose_verifier":
        us = ["s4b.verify.insar_PF.cells.2021_2022", "s4b.verify.wra_gnss.g1_recompute",
              "s4b.verify.amp_census_easting.yunlin", "s4b.verify.ref_coef_harmonics"]
        rec = prereg3.record_look(
            us, "insar_gnss_census_verification:2020-01..2022-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={
                us[0]: {"description": "verifier: InSAR-only comparison of runs P and F at "
                                       "active-cell pixels (changes within segments, across "
                                       "the 2022 gaps, a G1-like span)",
                        "cells": cells, "window": "2021-10..2022-12"},
                us[1]: {"description": "verifier: G1 recomputed from the evidence CSVs and "
                                       "rebuilt from cum.h5 and the WRA L0 files",
                        "ids": st_ids, "window": "2021-08..2022-12"},
                us[2]: {"description": "verifier: Y1 recomputed; rho(amplitude, easting); "
                                       "census density vs distance west of the pinch-out; "
                                       "count of flag-9 census cell-months",
                        "cells": yunlin_cells(), "window": "2020-01..2022-12"},
                us[3]: {"none": "harmonic fit of the run-P stable-area reference "
                                "coefficients: off-fan stable pixels only, no active cell, "
                                "well, benchmark or GNSS station"}},
            note="logged by the S4b repair round on behalf of the independent verifier of "
                 "2026-10-14, whose report disclosed these looks; the verifier logged nothing")
    elif a.key == "repair_gap_meta":
        u = "s4b.repair.network_gaps.cells"
        rec = prereg3.record_look(
            [u], "insar_licsbas_network_gaps:2020-01..2026-01", C.WP, "integrity_check",
            members={u: {"description": "LiCSBAS network structure of runs P, F and D: "
                                        "12network_gap_info, per-pixel gap intervals (cum.h5 "
                                        "'gap') and n_gap at the active cells",
                         "cells": cells, "window": "2020-01..2026-01"}},
            note="s4b_r2_gaps.py: which interferograms are valid per pixel and where the "
                 "network breaks; no displacement, coherence, GNSS or census value is read")
    else:
        us = ["s4b.repair.insar_PF.cells.2021_2022", "s4b.repair.wra_gnss.g1_pieces",
              "s4b.repair.amp_census_easting.yunlin", "s4b.repair.ref_coefs"]
        rec = prereg3.record_look(
            us, "insar_gnss_gap_diagnostics:2020-01..2025-12", C.WP, "after_the_fact",
            prereg_id=C.PREREG_ID,
            members={
                us[0]: {"description": "runs P and F at active-cell pixels and cells: change "
                                       "differences per interval class (within segments, "
                                       "across each 2022 gap, the G1 span)",
                        "cells": cells, "window": "2021-08..2022-12"},
                us[1]: {"description": "the G1 error of runs P and F at the 18 G1 stations "
                                       "split into cross-gap and within-segment increments; "
                                       "single-interval increment errors",
                        "ids": st_ids, "window": "2021-08..2022-12"},
                us[2]: {"description": "rho(amplitude, easting) and rho(census density, "
                                       "easting) on the Yunlin cells; flag-9 census "
                                       "cell-months 2020-22",
                        "cells": yunlin_cells(), "window": "2020-01..2022-12"},
                us[3]: {"none": "stable-area reference coefficients (off-fan stable pixels "
                                "only) of runs P (harmonics) and F (2020-2025 drifts)"}},
            note="S4b repair round: after-the-fact diagnostics of the registered G1 failure and "
                 "of the Y and ramp interpretations; report only, never a gate, threshold or "
                 "design choice")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units", "data_scope")
                      if k in rec}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

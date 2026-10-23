"""Write MANIFEST.json for results/dataprep/et0/v<N> (WP W2), after build_et0.py and
station_et0.py ran.

Usage: python manifest_w2.py --dir results/dataprep/et0/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archive_probes import DEST as VARIANT_DIR
from build_et0 import HISTORY, MAIN, PROBE, REFETCH, SPLICE
from common import manifest
from wcommon import ET_CACHE, FORCING_EXP_ET, GW_STATIONS, HERE, PREREG_DIR, REPO

PRE, POST = "2022-12", "2026-09"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir) if Path(a.dir).is_absolute() else REPO / a.dir
    v = json.loads((d / "validation.json").read_text())
    outputs = [{"path": "et0_daily_2012_2022_preorigin.npz", "fit_window_end": PRE},
               {"path": "et0_daily_2012_2026.npz", "fit_window_end": POST},
               {"path": "validation.json", "fit_window_end": POST},
               {"path": "climatology_bias_2023_2026.csv", "fit_window_end": POST},
               {"path": "station_check_467270_monthly.csv", "fit_window_end": POST},
               {"path": "station_check_467270.json", "fit_window_end": POST}]
    station_raw = REPO / "data_fetch_api/new_measurements/cwa_codis_rain/raw"
    st_files = sorted(p for p in station_raw.glob("467270_*.json")
                      if not p.name.endswith(".meta.json"))
    m = manifest.build_manifest(
        d, product="et0", layer="L1",
        inputs=[ET_CACHE, GW_STATIONS, REPO / "data_fetch_api/new_measurements/"
                "cwa_station_inventory/codis_station_list_2026-10-11.json",
                PREREG_DIR / "W2.et0_overlap.v1.json", PREREG_DIR / "W2.et0_overlap.v2.json",
                PREREG_DIR / "W2.et0_overlap.v3.json",
                *st_files],
        input_trees=[FORCING_EXP_ET, MAIN, REFETCH, HISTORY, PROBE, VARIANT_DIR],
        code=[HERE / "build_et0.py", HERE / "fetch_et0.py", HERE / "station_et0.py",
              HERE / "archive_probes.py",
              HERE / "wcommon.py", HERE / "prereg.py", HERE / "prereg_a1.py",
              HERE / "isorun.py", HERE / "manifest_w2.py"],
        parameters={"day_convention": "GMT (UTC) calendar days (the cache's verified "
                                      "convention)",
                    "splice": {str(k): list(x) for k, x in SPLICE.items()},
                    "level_match": "per-point additive offset mean(cache - forcing "
                                   "experiment) over 2022, applied from 2023-01-01 when the "
                                   "pre-registered stability rule holds; clipped at 0",
                    "primary_variant": v["primary_variant"],
                    "thresholds": {"overlap_mae_mm_d": 0.1, "completeness": 0.99,
                                   "level_min_mm_d": 0.02, "stability_tol_mm_d": 0.02},
                    "preregistration": "W2.et0_overlap.v2 (v1 superseded, never run); A1-tool "
                                       "record W2.et0_overlap.v3 (design unchanged)"},
        fit_window_end=POST, validation_report="validation.json",
        gate={"passed": bool(v["gate_passed"]),
              "criteria": "plan 3 WP W2 and pre-registration W2.et0_overlap.v2: T0 same grid "
                          "cells, T1 2022 overlap max per-point MAE < 0.1 mm/d, T2 chain "
                          "overlap, T3 pre-freeze re-fetch, >= 99 % finite days 2023-26"},
        allowed_uses=["forcing_recorded_arm", "forcing_frozen", "statistical_layer_history",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                        "prospective_scorer", "selection", "model_design", "prior",
                        "prior_width_input", "initial_conditions", "idw_field"],
        personal_data="none (public weather reanalysis at well-station coordinates; station "
                      "467270 public meteorology)",
        outputs=outputs,
        extra={"notes": [
            "forcing_frozen (pre-origin) may read only et0_daily_2012_2022_preorigin.npz, "
            "which holds the calibration cache values unchanged (GMT days, 2012-2022).",
            "2023-26 ET0 is a post-origin forcing (U4): recorded-forcing diagnostic arm only.",
            "Primary 'et0' in et0_daily_2012_2026.npz is level-matched from 2023-01-01 (flag "
            "1) when validation.json says so; 'et0_raw_splice' keeps the source values.",
            "source 3 days (2026-07-01..09-30) are provisional: recent archive days may be "
            "revised; re-fetch before relying on them after 2027-01.",
            "Values are reanalysis-derived (ECMWF IFS via the Open-Meteo default model), not "
            "measurements: flag 0 means 'source value as served, not filled'."]})
    p = manifest.write_manifest(d, m)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

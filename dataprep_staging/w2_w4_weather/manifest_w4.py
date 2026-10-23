"""Write MANIFEST.json for results/dataprep/rain_check/v<N> (WP W4), after rain_check.py ran.

Usage: python manifest_w4.py --dir results/dataprep/rain_check/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wcommon import CODIS_STATION_LIST, HERE, REPO, RF_STATIONS, RF_TIMESERIES

from common import manifest

PRE = "2022-12"
POST = "2026-09"
PRE_OUTPUTS = ["validation.json", "logo_predictions_2012_2022.csv",
               "logo_per_gauge_2012_2022.csv", "variograms_2012_2022.json"]
POST_OUTPUTS = ["logo_predictions_2023_2026.csv", "logo_per_gauge_2023_2026.csv",
                "secondary_2023_2026.json", "residual_gaps_2012_2026.csv"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir) if Path(a.dir).is_absolute() else REPO / a.dir
    val = json.loads((d / "validation.json").read_text())
    dec = val["decision"]
    outputs = ([{"path": p, "fit_window_end": PRE} for p in PRE_OUTPUTS]
               + [{"path": p, "fit_window_end": POST} for p in POST_OUTPUTS])
    m = manifest.build_manifest(
        d, product="rain_check", layer="L2",
        inputs=[RF_TIMESERIES, RF_STATIONS, CODIS_STATION_LIST,
                REPO / "results/dataprep/preregistration/W4.rain_kriging_vs_idw.v1.json"],
        code=[HERE / "rain_check.py", HERE / "wcommon.py", HERE / "manifest_w4.py",
              REPO / "hydrophysics/subsidence.py"],
        parameters={"windows": {"primary_decision": ["2012-01", PRE],
                                "secondary_after_the_fact": ["2023-01", POST]},
                    "monthly_rule": "kept iff finite days >= 80 % of the month; total = mean "
                                    "x days", "min_other_gauges": 10, "idw_power": 2.0,
                    "variogram": "exponential + nugget, standardised OLS residuals, 2.5 km "
                                 "bins on 0-40 km, WLS pair-count weights, refit per held-out "
                                 "gauge on the primary window",
                    "drift": "altitude (m) of rf_stations.rf_num in the CODIS station list",
                    "bootstrap": {"n": 2000, "block_months": 12, "seed": 2026},
                    "preregistration": "W4.rain_kriging_vs_idw.v1"},
        fit_window_end=POST, validation_report="validation.json",
        gate={"passed": True,
              "criteria": "plan 3 WP W4 and pre-registration W4.rain_kriging_vs_idw.v1: the "
                          "leave-one-gauge-out comparison ran as registered and the adoption "
                          f"rule was applied ({dec['outcome']}). 'passed' means the report is "
                          "complete; it does not mean kriging passed."},
        allowed_uses=["model_design", "design_time_evidence", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                        "prospective_scorer", "forcing_frozen", "forcing_recorded_arm",
                        "selection", "prior", "prior_width_input", "initial_conditions",
                        "idw_field"],
        personal_data="none (public CWA gauge ids, coordinates and altitudes only)",
        outputs=outputs,
        extra={"decision": dec,
               "notes": ["Report only (plan W4): KED was not adopted, so no rainfall field "
                         "is produced; the twin keeps masked IDW.",
                         "Pre-origin outputs (fit_window_end 2022-12) may inform model "
                         "design; the 2023-26 outputs and the residual-gap table are "
                         "after-the-fact analysis only.",
                         "Ordinary kriging is a registered secondary diagnostic and cannot "
                         "be adopted from these units (post-hoc selection)."]})
    p = manifest.write_manifest(d, m)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

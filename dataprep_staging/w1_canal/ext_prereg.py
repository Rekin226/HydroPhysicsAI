"""Pre-registration of the W1 2026 extension (plan W1 step 6), via the A1 tool.

    python -I dataprep_staging/w1_canal/ext_prereg.py

Written before any rainfall-runoff regression or chain was scored. Two parts:

* the PROSPECTIVE test (fresh units): when the 2026 yearbook appears (about mid-2027), the
  monthly composite inflow Jan-Sep 2026 predicted from basin rain is scored against the
  yearbook composite, against the composite's monthly climatology 2012-2025;
* the SHIPPING RULE evaluated now on design-time units: the leave-one-year-out chain test
  over 2012-2025 (the Jiji annual district totals were fit targets of beta in full/v1, so
  these units are not fresh; the test decides only which 2026 series ships and its label).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.paths import rel_to_repo  # noqa: E402
from harness import prereg  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402

PREREG_ID = "W1.ext2026_rain_runoff.v1"
TEST_ID = "W1.ext2026.prospective_inflow_2026"
FULL_V1 = C.product_dir_base("full") / "v1"


def doc() -> dict:
    na = prereg.not_applicable
    months = [f"2026-{m:02d}" for m in range(1, 10)]
    return {
        "id": PREREG_ID, "wp": "W1", "kind": "test",
        "title": "W1 step 6: 2026 canal deliveries from a rainfall-runoff substitute for the "
                 "yearbook inflow composite",
        "tests": [{
            "test_id": TEST_ID, "data_scope": "wra_hydro_yearbook:2026-01..2026-09",
            "fresh_units": [f"W1.inflow.{m}" for m in months],
            "metric": "RMSE over Jan-Sep 2026 of the monthly composite volume (10^8 m3, "
                      "H075+H049+H064 rescaled to the weir's 2026 annual inflow as in "
                      "full/v1) predicted from the basin rain index",
            "baseline": "monthly climatology of the composite volume 2012-2025 (full/v1)",
            "pass_if": "RMSE(prediction) < RMSE(climatology); reported with a sign test "
                       "over the 9 months",
            "when": "after the 2026 Hydrological Year Book and the 2026 Jiji annual table "
                    "are published (expected mid-2027); the prediction is the one shipped "
                    "in surface_water_v4/full/v2 (inflow_predicted_monthly.csv)"}],
        "zones": na("district-level series (Changhua, Yunlin) and one basin index; no "
                    "spatial zones"),
        "priors": na("ordinary least squares regression; no priors"),
        "burn_in": na("no dynamic model is started"),
        "folds": {"design_time": "leave-one-year-out over 2012-2025 with nested leave-one-"
                                 "year-out selection of the regression spec on the training "
                                 "years; scored on the evaluation years (every month from "
                                 "November of the previous year to December has a rain "
                                 "index; CODIS has 2-7 of 16 records mid-2014..2015)",
                  "prospective": "single fold: 2026-01..2026-09"},
        "fresh_units": [f"W1.inflow.{m}" for m in months],
        "thresholds": {
            "shipping_rule": "the rain-driven chain ships for 2026 if its leave-one-year-out "
                             "annual RMSE (district deliveries, 10^8 m3) is below annual "
                             "climatology (mean of the other years) in both districts; label "
                             "'beats climatology' if both 90 % paired-bootstrap intervals of "
                             "the RMSE difference are below zero, else 'not distinguishable "
                             "from climatology'; if either point RMSE is not below, the 2026 "
                             "months ship as the monthly climatology of the full/v1 Denton "
                             "series 2012-2025",
            "bootstrap": {"B": E.BOOT_B, "seed": E.BOOT_SEED, "level": E.BOOT_LEVEL},
            "design_time_units": "W1.annual.{chc,yl}.<evaluation years> (not fresh: fit "
                                 "targets of beta in full/v1)",
            "sd_of_2026_months": "per district and calendar month, RMSE over the scored folds of "
                                 "(chain - full/v1 Denton); structural-zero months take the "
                                 "mean full/v1 sd of that calendar month",
            "draws": f"{E.N_DRAWS} draws: block bootstrap of whole-year fold residuals of "
                     f"the scored folds (chain - Denton, additive, clipped at 0; one residual "
                     f"year per member for both districts), seed {E.DRAW_SEED}"},
        "gate_scores": [{"name": "annual_loo_rmse_rain_chain_minus_climatology",
                         "weight": 1.0, "districts": ["chc", "yl"]}],
        "primary_model": {
            "rain_index": "16 CODIS records in the drainage areas of H075/H049/H064 "
                          "(ext_config.STATIONS), normal-ratio averaging (calendar-month "
                          "normals 2012-2022), area-weighted sub-basins, months with >= 8 "
                          "valid records",
            "regression": "monthly composite volume on rain index lags 0-2 with calendar-"
                          "month intercepts; linear or log-log chosen by nested leave-one-"
                          "year-out RMSE",
            "chain": "diversion.simulate with (beta, a_chc, a_yl) refitted on the training "
                     "years, full/v1 shares (0.5/0.3) and winter weight (0), R = previous "
                     "year's industrial + public supply (2026: R of 2025)",
            "flag": "6 (extrapolated beyond the record end; never calibration)",
            "use": "surface_water_v4/full/v2 only (U4: recorded-forcing arm)"},
        "inputs": [rel_to_repo(p) for p in (
            HERE / "ext_config.py", HERE / "ext_model.py", HERE / "ext_build.py",
            HERE / "diversion.py", E.L0_DIR / "SOURCE.json",
            FULL_V1 / "MANIFEST.json", FULL_V1 / "inflow_composite_daily.parquet",
            FULL_V1 / "district_monthly.csv", FULL_V1 / "validation.json", C.JIJI_CSV,
            C.L0_NEW / "cwa_station_inventory" / "codis_station_list_2026-10-11.json")],
        "interpretation": (
            "The design-time chain test only chooses between the rain-driven 2026 series "
            "and climatology and sets its label, sd and draws; it is not evidence of skill "
            "on fresh data, because the annual totals were fit targets of full/v1 and the "
            "shares were chosen on the 2015-2021 Jan-May benchmark. Skill claims for 2026 "
            "wait for the prospective test. The 2026 months are post-origin forcing (U4): "
            "recorded-forcing arm only, flag 6, never calibration or scoring."),
    }


if __name__ == "__main__":
    path, sha = prereg.write_prereg(doc())
    print(path, sha)

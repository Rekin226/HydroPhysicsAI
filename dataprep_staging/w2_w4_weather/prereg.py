"""Pre-registrations for WP W2 and WP W4, written BEFORE the registered tests run (plan 4.5).

The WP A1 pre-registration tool does not exist yet (A1 has not run), so this script writes
the hashed JSON itself to results/dataprep/preregistration/<id>.json, refusing to overwrite,
and prints each file's sha256 (recorded in the W2W4 ledger fragment and report).

Usage: python prereg.py {W2,W4}
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from wcommon import (
    CODIS_STATION_LIST,
    ET_CACHE,
    FORCING_EXP_ET,
    GW_STATIONS,
    OM_ARCHIVE,
    OM_VAR,
    PREREG_DIR,
    REPO,
    RF_STATIONS,
    RF_TIMESERIES,
    sha256_file,
    write_json_new,
)


def _rel(p):
    return p.relative_to(REPO).as_posix()


def w2() -> dict:
    return {
        "id": "W2.et0_overlap.v1",
        "wp": "W2",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP W2",
        "question": "Is a local-day Open-Meteo archive fetch of daily FAO-56 ET0 at the 61 "
                    "calibration points consistent with the calibration cache in their "
                    "2022 overlap, so that 2023-01..2026-09 can be spliced onto the cache?",
        "inputs": {
            "cache": {"path": _rel(ET_CACHE), "sha256": sha256_file(ET_CACHE)},
            "points": {"path": _rel(GW_STATIONS), "sha256": sha256_file(GW_STATIONS),
                       "rule": "st_id with TM_X97/TM_Y97 (EPSG:3826) -> EPSG:4326, rounded to "
                               "5 decimals"},
            "forcing_experiment_gmt": {"path": _rel(FORCING_EXP_ET),
                                       "use": "secondary: GMT vs local day convention"},
        },
        "fetch": {"url": OM_ARCHIVE, "daily": OM_VAR, "timezone": "Asia/Taipei",
                  "start_date": "2022-01-01", "end_date": "2026-09-30",
                  "models": "API default (no models parameter), as the cache and the "
                            "forcing experiment", "one request per point": True,
                  "refetch_before_freeze": "2026-07-01..2026-09-30 for all 61 points, at "
                                           "least 60 min after the main fetch ends"},
        "primary_test": {
            "metric": "per-point daily MAE |new - cache| over 2022-01-01..2022-12-31, days "
                      "finite in both",
            "criterion": "max over the 61 points of the per-point MAE < 0.1 mm/d (plan W2 "
                         "threshold, read strictly: every point)",
        },
        "gate": "primary_test passes AND every point has >= 99 % finite days over "
                "2023-01-01..2026-09-30 (missing days stay NaN with flag 8, never filled)",
        "splice_rule": "pass: 2012-01-01..2022-12-31 = cache values unchanged (source 0); "
                       "2023-01-01..2026-06-30 = main fetch (source 1); "
                       "2026-07-01..2026-09-30 = pre-freeze re-fetch (source 2, provisional). "
                       "fail: no splice; et0/v1 gate fails and the report says so.",
        "secondary_reported_not_gating": [
            "pooled and median per-point MAE, bias and Pearson r in 2022",
            "new local fetch vs forcing-experiment GMT fetch (2022-01..2025-06): MAE, i.e. the "
            "day-convention effect",
            "fan-mean monthly ET0 2023-01..2026-09 minus the 2012-2021 cache climatology of the "
            "same calendar month (mm/d), with z = bias / interannual sd (2012-2021)",
            "revision check: re-fetch vs main fetch on 2026-07-01..2026-09-30: max |diff| and "
            "number of changed values",
            "grid cell and elevation returned per point vs the forcing experiment's responses",
        ],
        "anti_leakage": "No fitted parameter. 2023+ ET0 is a post-origin forcing (U4): "
                        "forcing_recorded_arm only. The pre-origin output (2012-2022, cache "
                        "values) carries fit_window_end 2022-12.",
    }


def w2v2() -> dict:
    return {
        "id": "W2.et0_overlap.v2",
        "wp": "W2",
        "supersedes": {
            "id": "W2.et0_overlap.v1",
            "status_of_superseded": "never run (no 61-point fetch was made under it)",
            "reason": "v1 assumed, from the plan, that the calibration cache uses local "
                      "(Asia/Taipei) days. A one-point probe (st1, 2022, local days, "
                      "openmeteo_et0/w2_probe_st1_2022) and the forcing experiment's GMT fetch "
                      "show the cache is in GMT days: r(cache, GMT) = 0.9998 at lag 0 for st1 "
                      "and >= 0.9996 for all 61 points (r = 0.70 at +-1 day), r(cache, local) "
                      "= 0.983, daily MAE 0.150 against local days. The 2022 difference against "
                      "GMT (MAE 0.049) is a near-constant additive offset (cache lower; OLS "
                      "slope 1.001), not a day-convention effect.",
            "design_time_evidence": ["data_fetch_api/new_measurements/openmeteo_et0/"
                                     "w2_probe_st1_2022/st1.json",
                                     "data_fetch_api/forcing_experiment/et (2022 rows)",
                                     "January 2022 endpoint/model/timezone variants at st1 "
                                     "(scratch only): /v1/era5 = /v1/archive = models=ecmwf_ifs"
                                     " (the API default); era5, era5_land, era5_seamless do "
                                     "not reproduce the cache"],
        },
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP W2",
        "day_convention": "GMT (UTC) calendar days, the cache's verified convention (the plan's "
                          "intent is 'one convention, matching the cache')",
        "inputs": {
            "cache": {"path": _rel(ET_CACHE), "sha256": sha256_file(ET_CACHE),
                      "span": "2012-01-01..2022-12-31", "source_code": 0},
            "forcing_experiment_gmt": {"path": _rel(FORCING_EXP_ET),
                                       "span": "2022-01-01..2025-06-30 (fetched 2026-09-30)",
                                       "source_code": 1},
            "points": {"path": _rel(GW_STATIONS), "sha256": sha256_file(GW_STATIONS)},
        },
        "new_fetches": {
            "main": {"url": OM_ARCHIVE, "daily": OM_VAR, "timezone": "GMT",
                     "start_date": "2025-01-01", "end_date": "2026-09-30",
                     "points": "all 61", "source_code": 2},
            "refetch_before_freeze": {"timezone": "GMT", "start_date": "2026-07-01",
                                      "end_date": "2026-09-30", "points": "all 61",
                                      "when": ">= 60 min after the main fetch ends",
                                      "source_code": 3, "status": "provisional"},
            "offset_history": {"timezone": "GMT", "start_date": "2012-01-01",
                               "end_date": "2021-12-31",
                               "points": ["st1 (north)", "st42 (southernmost)"]},
        },
        "splice": "2012-01-01..2022-12-31 cache (0); 2023-01-01..2025-06-30 forcing "
                  "experiment (1); 2025-07-01..2026-06-30 main fetch (2); 2026-07-01.."
                  "2026-09-30 re-fetch (3)",
        "tests": {
            "T0_same_points": "each point's returned grid latitude, longitude and elevation in "
                              "the main fetch equal the forcing experiment's (reported; a "
                              "mismatch fails the gate)",
            "T1_primary_plan_acceptance": "2022 overlap, cache vs forcing-experiment GMT: "
                                          "per-point daily MAE < 0.1 mm/d for every point. "
                                          "Disclosed: the plan already computed this (median "
                                          "0.048, max 0.063), so it is a re-computation, not a "
                                          "fresh test.",
            "T2_chain": "2025-01-01..2025-06-30, main fetch vs forcing experiment: per-point "
                        "MAE < 0.1 mm/d for every point",
            "T3_revision": "2026-07-01..2026-09-30, re-fetch vs main fetch: per-point MAE < "
                           "0.1 mm/d for every point; max |diff| and changed-value count "
                           "reported",
            "completeness": ">= 99 % finite days per point over 2023-01-01..2026-09-30; "
                            "missing days stay NaN, flag 8",
        },
        "gate": "T0, T1, T2, T3 and completeness all pass",
        "level_match_rule": {
            "offset": "per point delta_p = mean(cache - forcing-experiment GMT) over 2022 "
                      "(pre-origin overlap), additive",
            "stability": "annual mean (cache - offset_history fetch) for 2012..2021 at st1 and "
                         "st42",
            "rule": "if |fan-mean delta| >= 0.02 mm/d AND every annual offset 2012-2021 at "
                    "both points lies within +-0.02 mm/d of that point's 2022 offset (computed "
                    "the same way from the forcing experiment), the primary array 'et0' holds "
                    "level-matched values from 2023-01-01 on (value + delta_p, flag 1, method "
                    "level_match_2022_additive); otherwise 'et0' holds the raw splice (flag 0) "
                    "and the offsets are only reported. Both variants are always written "
                    "('et0_raw_splice', 'et0_level_matched').",
        },
        "secondary_reported_not_gating": [
            "fan-mean monthly ET0 2023-01..2026-09 minus the 2012-2021 cache climatology of the "
            "same calendar month (mm/d) and z = bias / interannual sd, for both variants",
            "local-day vs GMT effect at st1 in 2022 (probe)",
            "multiplicative alternative to the additive offset (RMSE after each)",
        ],
        "anti_leakage": "Level-match offsets are fitted on 2022 (pre-origin) only. 2023+ ET0 is "
                        "a post-origin forcing (U4): forcing_recorded_arm only. The 2012-2022 "
                        "output keeps the cache values with fit_window_end 2022-12.",
    }


def w4() -> dict:
    return {
        "id": "W4.rain_kriging_vs_idw.v1",
        "wp": "W4",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP W4",
        "question": "Does kriging with an elevation drift (KED) beat the twin's masked IDW at "
                    "predicting monthly gauge rainfall totals, leave-one-gauge-out?",
        "inputs": {
            "rain": {"path": _rel(RF_TIMESERIES), "sha256": sha256_file(RF_TIMESERIES),
                     "note": "CWA CODIS rebuild installed as the twin's rf_timeseries.csv"},
            "gauges": {"path": _rel(RF_STATIONS), "sha256": sha256_file(RF_STATIONS),
                       "rule": "TM_X97/TM_Y97 (m), the coordinates the twin's IDW uses"},
            "altitude": {"path": _rel(CODIS_STATION_LIST),
                         "sha256": sha256_file(CODIS_STATION_LIST),
                         "rule": "altitude (m) of the station named by rf_stations.rf_num; "
                                 "time-invariant"},
        },
        "monthly_totals": "twin rule (calibrate_flow.monthly_mean_rate): a gauge-month is kept "
                          "iff finite days >= 80 % of the month's days; total = mean of the "
                          "finite days x days in month (mm/month)",
        "windows": {"primary_decision": ["2012-01", "2022-12"],
                    "secondary_after_the_fact": ["2023-01", "2026-09"],
                    "rule": "only the primary (pre-origin) window decides (U3); the secondary "
                            "window uses variograms fitted on the primary window and is "
                            "reported as after-the-fact analysis only"},
        "logo": "for each gauge g and month m with g kept: predict g from every other gauge "
                "kept in m; the gauge-month enters both methods only if >= 10 other gauges are "
                "kept in m",
        "incumbent": {"name": "masked IDW",
                      "definition": "hydrophysics.subsidence.idw_interp, power 2, eps 1e-6 "
                                    "m^2, missing gauges dropped per month (the twin's "
                                    "_idw_field)"},
        "candidate": {
            "name": "KED",
            "definition": "universal kriging with drift terms [1, altitude]; residual "
                          "variogram gamma(h) = c0 + c1 (1 - exp(-h/a)), gamma(0) = 0",
            "variogram_fit": "per held-out gauge (refit on the 25 training gauges), primary "
                             "window months with >= 10 training gauges kept: OLS residuals of "
                             "the totals on [1, altitude], divided by their sd (ddof 2; months "
                             "with sd < 1e-6 skipped); all pairs pooled; lag bins 2.5 km wide "
                             "on 0-40 km; weighted least squares (weights = pair counts) for "
                             "c0 >= 0, c1 >= 0, a in [0.5, 200] km",
            "postprocess": "predictions clipped at 0 mm",
            "fallback": "a singular system is solved by least squares; the count is reported",
        },
        "metrics": "fan-wide RMSE over all LOGO gauge-months (mm/month); per-gauge RMSE",
        "decision_rule": "adopt KED iff RMSE_KED <= 0.90 x RMSE_IDW fan-wide AND, for every "
                         "gauge, RMSE_KED,g <= 1.20 x RMSE_IDW,g (plan W4). Otherwise masked "
                         "IDW stays and rain_check/v1 is a report only.",
        "secondary_reported_not_adoptable": [
            "ordinary kriging (same variogram procedure, drift [1] only)",
            "bias, wet (May-Oct) vs dry (Nov-Apr) season RMSE",
            "paired moving-block bootstrap of the fan-wide RMSE ratio (resample months, 12-month "
            "blocks, 2000 replicates, seed 2026)",
            "secondary window 2023-01..2026-09 (after the fact)",
            "residual gaps: kept gauge-months per gauge and per month",
        ],
        "if_adopted": "rain_check/v1 adds the KED monthly field on the twin grid with a "
                      "kriging sd whose 90 % LOGO coverage is checked before it is quoted",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["W2", "W2v2", "W4"])
    a = ap.parse_args(argv)
    rec = {"W2": w2, "W2v2": w2v2, "W4": w4}[a.which]()
    rec["created_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["written_by"] = "dataprep_staging/w2_w4_weather/prereg.py (A1 tool not available yet)"
    p = write_json_new(PREREG_DIR / f"{rec['id']}.json", rec, mode=0o444)
    print(p, sha256_file(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())

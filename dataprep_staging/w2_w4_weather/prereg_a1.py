"""WP W2W4 pre-registration records written with the WP A1 tool (harness.prereg).

The first W2W4 round (2026-10-12, 02:30-02:49Z) wrote its pre-registrations with this WP's
own ``prereg.py`` because the A1 tool did not exist yet (harness/prereg.py appeared at about
02:42Z). This script brings them under the A1 tool without back-dating anything:

* ``W2v3``: a kind "test" pre-registration ``W2.et0_overlap.v3`` that adopts the design of
  ``W2.et0_overlap.v2`` unchanged (v2 is a hashed input) for the tests that have NOT been
  scored yet: T0 (same grid cells), T2 (chain overlap 2025-01..06), T3 (pre-freeze
  re-fetch 2026-07..09) and completeness 2023-01..2026-09. It is written before the main
  fetch finishes, so before any of them can be computed. ``build_et0.py`` calls
  ``assert_can_score`` on it before scoring, which logs the look.
* ``W4look``: the W4 leave-one-gauge-out test ran at about 02:37Z under the WP-written
  pre-registration ``W4.rain_kriging_vs_idw.v1`` (02:30:47Z). A test pre-registration
  written now would postdate that look, so instead this records (a) a kind "declaration"
  that puts W4 v1's sha256 into the hash-chained INDEX and (b) the scoring looks on the 26
  gauges for both windows, with the real time in the note, so the units can never be
  reused as fresh units for a later rain-interpolation test.

Usage: python -I isorun.py prereg_a1.py {W2v3,W4look}
"""
from __future__ import annotations

import argparse
import json
import sys

from harness import prereg as P
from wcommon import (
    CODIS_STATION_LIST,
    ET_CACHE,
    GW_STATIONS,
    PREREG_DIR,
    RF_STATIONS,
    RF_TIMESERIES,
    sha256_file,
)

WP = "W2W4"
NA = P.not_applicable

UNIT_MAIN_2025H1 = "openmeteo_et0_main_2025-01..2025-06_61pts"
UNIT_REFETCH = "openmeteo_et0_refetch_2026-07..2026-09_61pts"
UNIT_CELLS = "openmeteo_et0_main_vs_fe_gridcells_61pts"
UNIT_COMPLETE = "et0_splice_2023-01..2026-09_61pts"


def w2v3() -> dict:
    v2 = PREREG_DIR / "W2.et0_overlap.v2.json"
    v1 = PREREG_DIR / "W2.et0_overlap.v1.json"
    ext = {UNIT_MAIN_2025H1: "new fetch openmeteo_et0/w2_gmt_2025_2026, days 2025-01-01.."
                             "2025-06-30, all 61 points, compared with the forcing-experiment "
                             "fetch of 2026-09-30",
           UNIT_REFETCH: "pre-freeze re-fetch openmeteo_et0/w2_gmt_refetch_2026q3 (not yet "
                         "fetched), days 2026-07-01..2026-09-30, compared with the main fetch",
           UNIT_CELLS: "grid latitude, longitude and elevation returned per point by the main "
                       "fetch and by the forcing experiment",
           UNIT_COMPLETE: "finite-day fraction per point of the spliced product, 2023-01-01.."
                          "2026-09-30"}
    return {
        "id": "W2.et0_overlap.v3",
        "wp": WP,
        "kind": "test",
        "title": "W2 ET0 extension: integrity tests of the 2023-26 splice (A1-tool record of "
                 "W2.et0_overlap.v2, design unchanged)",
        "adopts": {"id": "W2.et0_overlap.v2", "path": "results/dataprep/preregistration/"
                   "W2.et0_overlap.v2.json", "sha256": sha256_file(v2),
                   "created_utc": "2026-10-12T02:35:37Z",
                   "note": "v2 was written by dataprep_staging/w2_w4_weather/prereg.py before "
                           "the A1 tool existed. Every rule, threshold, splice and the "
                           "level-match rule are taken from v2 without change; v3 only adds "
                           "the A1 index, test ids and the look log."},
        "tests": [
            {"test_id": "W2.v3.T0_same_points", "data_scope": "openmeteo_et0:gridcells",
             "fresh_units": [UNIT_CELLS], "external_units": {UNIT_CELLS: ext[UNIT_CELLS]}},
            {"test_id": "W2.v3.T2_chain_2025H1", "data_scope": "openmeteo_et0:2025-01..2025-06",
             "fresh_units": [UNIT_MAIN_2025H1],
             "external_units": {UNIT_MAIN_2025H1: ext[UNIT_MAIN_2025H1]}},
            {"test_id": "W2.v3.T3_refetch_2026Q3",
             "data_scope": "openmeteo_et0:2026-07..2026-09", "fresh_units": [UNIT_REFETCH],
             "external_units": {UNIT_REFETCH: ext[UNIT_REFETCH]}},
            {"test_id": "W2.v3.completeness_2023_2026",
             "data_scope": "et0_splice:2023-01..2026-09", "fresh_units": [UNIT_COMPLETE],
             "external_units": {UNIT_COMPLETE: ext[UNIT_COMPLETE]}},
        ],
        "zones": NA("61 point series; no zones"),
        "priors": NA("no fitted parameter in the tests"),
        "burn_in": NA("no model run"),
        "folds": NA("no folds; every point is checked"),
        "fresh_units": [UNIT_CELLS, UNIT_MAIN_2025H1, UNIT_REFETCH, UNIT_COMPLETE],
        "thresholds": {
            "T0": "0 points whose returned (latitude, longitude, elevation) differ between the "
                  "main fetch and the forcing experiment (pass/fail)",
            "T2": "per-point daily MAE |main - forcing experiment| over 2025-01-01..2025-06-30 "
                  "< 0.1 mm/d for every one of the 61 points (pass/fail)",
            "T3": "per-point daily MAE |re-fetch - main| over 2026-07-01..2026-09-30 < 0.1 "
                  "mm/d for every point; max |diff| and changed-value count reported "
                  "(pass/fail); the re-fetch runs >= 60 min after the main fetch ends",
            "completeness": ">= 99 % finite days per point over 2023-01-01..2026-09-30 in the "
                            "splice; missing days stay NaN with flag 8 (pass/fail)",
            "gate": "et0/v<N> gate = T0 and T1 and T2 and T3 and completeness (T1 = 2022 "
                    "overlap cache vs forcing experiment, per-point MAE < 0.1 mm/d, the plan's "
                    "acceptance; a disclosed re-computation, see prior_exposure)",
            "level_match": "from v2 unchanged: per-point additive offset delta_p = mean(cache "
                           "- forcing experiment) over 2022; applied to 'et0' from 2023-01-01 "
                           "(flag 1) iff |fan mean delta| >= 0.02 mm/d and every annual offset "
                           "2012-2021 at st1 and st42 lies within +-0.02 mm/d of that point's "
                           "2022 offset; both variants always written",
        },
        "gate_scores": NA("the gate is the conjunction of pass/fail tests; no weighted score"),
        "primary_model": "et0/v<N>: splice 2012-2022 cache (0), 2023-01..2025-06 forcing "
                         "experiment (1), 2025-07..2026-06 main fetch (2), 2026-07..09 re-fetch "
                         "(3, provisional); GMT days",
        "inputs": [str(v2), str(v1), str(ET_CACHE), str(GW_STATIONS)],
        "interpretation": "A passed gate means the 2023-26 days come from the same Open-Meteo "
                          "series, cells and day convention as the 2022 overlap, are complete, "
                          "and were stable between the main fetch and the re-fetch. It does not "
                          "say which archive level (the June 2026 cache or today's archive) is "
                          "closer to the truth; the station check is post hoc and not gating. "
                          "2023+ ET0 is a post-origin forcing (U4): forcing_recorded_arm only.",
        "prior_exposure": [
            "T1 (2022 cache vs forcing experiment) was computed by the plan (median 0.048, max "
            "0.063 mm/d) and in the first W2W4 round; it is a re-computation, not a fresh test, "
            "and is therefore not listed as a test here.",
            "The first round (interrupted at 02:49Z) downloaded 18 of the 61 main-fetch "
            "responses (st1..st18). build_et0.py refuses to run with fewer than 61 points, and "
            "no T0/T2 result exists in the first round's files; a scratch look at those 18 "
            "points cannot be excluded and is disclosed.",
            "The first round fetched the 2012-2021 offset history at st1 and st42 (02:38-02:40Z,"
            " after v2) and its report draft states that the level-match rule held; the rule "
            "itself was fixed in v2 before that fetch.",
            "Open-Meteo endpoint/model/timezone variants at st1 (January 2022) and the one-point "
            "local-day probe were fetched before v2 and are design-time evidence for the GMT "
            "convention (listed in v2).",
        ],
        "external_units": ext,
    }


def w4_declaration() -> dict:
    v1 = PREREG_DIR / "W4.rain_kriging_vs_idw.v1.json"
    return {
        "id": "W4.rain_kriging_vs_idw.v1.record",
        "wp": WP,
        "kind": "declaration",
        "title": "Record of the WP-written W4 pre-registration (KED vs masked IDW, LOGO) and of "
                 "its already-made scoring look",
        "records": {"id": "W4.rain_kriging_vs_idw.v1", "path": "results/dataprep/"
                    "preregistration/W4.rain_kriging_vs_idw.v1.json", "sha256": sha256_file(v1),
                    "created_utc": "2026-10-12T02:30:47Z",
                    "scored_utc_approx": "2026-10-12T02:37Z (results/dataprep/rain_check/v1 "
                                         "file times)",
                    "outcome": "masked IDW stays (KED not adopted)"},
        "tests": NA("declaration: the test already ran under W4.rain_kriging_vs_idw.v1"),
        "zones": NA("26 gauges, fan-wide"),
        "priors": NA("no prior"),
        "burn_in": NA("no model run"),
        "folds": "leave-one-gauge-out over the 26 rf gauges (as registered in v1)",
        "fresh_units": NA("declaration; the 26 gauges are now looked-at units for rain "
                          "interpolation tests (see looks.jsonl)"),
        "thresholds": "from v1: adopt KED iff RMSE_KED <= 0.90 RMSE_IDW fan-wide and every "
                      "gauge RMSE_KED,g <= 1.20 RMSE_IDW,g, primary window 2012-01..2022-12",
        "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": [str(v1), str(RF_TIMESERIES), str(RF_STATIONS), str(CODIS_STATION_LIST)],
        "interpretation": "Puts v1's sha256 into the hash-chained index. Any later "
                          "rain-interpolation test (for example ordinary kriging, which looked "
                          "better as a non-adoptable secondary) must be pre-registered on units "
                          "other than these 26 gauges in 2012-2026.",
    }


GAUGES = [f"rf{i}" for i in range(1, 27)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["W2v3", "W4look"])
    a = ap.parse_args(argv)
    if a.which == "W2v3":
        path, sha = P.write_prereg(w2v3())
        print(json.dumps({"path": str(path), "sha256": sha}))
        return 0
    path, sha = P.write_prereg(w4_declaration())
    units = [f"rain_gauge_{g}" for g in GAUGES]
    looks = []
    for scope, when in (("rain_monthly:2012-01..2022-12", "primary window, decision"),
                        ("rain_monthly:2023-01..2026-09", "secondary window, after the fact")):
        looks.append(P.record_look(
            units, scope, WP, "score", test_id="W4.v1.logo_ked_vs_idw",
            prereg_id="W4.rain_kriging_vs_idw.v1",
            note=f"logged late: the LOGO scores ({when}) of masked IDW, KED and OK were "
                 "computed at about 2026-10-12T02:37Z under the WP-written pre-registration "
                 "W4.rain_kriging_vs_idw.v1 (02:30:47Z), before the A1 tool existed"))
    print(json.dumps({"path": str(path), "sha256": sha, "looks": len(looks)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

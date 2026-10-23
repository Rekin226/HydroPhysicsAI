"""WP S2 round 2: register the fresh test C with the A1 tool, before it is scored.

Test C compares the deep-benchmark feeds, as the product processes them, with the printed
table 1 of the WRA Changhua-Yunlin monthly data-analysis reports (series 007001,
``results/dataprep/deep_benchmarks_printed/v1``). No printed value of that series has ever
been compared with a feed in this project; the units are pile-months 2019-05..2023-12, before
the integrated reports start (2024-01) and therefore outside the design-time units of tests A
and B. Two tests: C1 on the pre-origin months (<= 2022-12, the months S1 uses), C2 on 2023.

Usage: python -I prereg_c.py [--dry]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, L0_NEW, PREREG_DIR, REPO, rel_to_repo  # noqa: E402
from harness import prereg as pr  # noqa: E402

from s2_deep_benchmarks import c_test as T  # noqa: E402

PREREG_ID = "S2r2.c_fresh_cy007001"
PRINTED = DATAPREP / "deep_benchmarks_printed" / "v1"
SENSORS = ["SJES", "TKJS(1)", "TKJS(2)", "TKJS(3)", "NTUH(1)", "NTUH(2)", "STES"]
C1 = {"test_id": "S2r2.C1_cy007001_preorigin",
      "data_scope": "deep_piles_cy007001_printed_vs_feed:2019-01..2022-12",
      "fresh_units": [f"dbm_cy007001_printed:{s}:pre" for s in SENSORS],
      "months": ["2019-01", "2022-12"]}
C2 = {"test_id": "S2r2.C2_cy007001_2023",
      "data_scope": "deep_piles_cy007001_printed_vs_feed:2023-01..2023-12",
      "fresh_units": [f"dbm_cy007001_printed:{s}:2023" for s in SENSORS],
      "months": ["2023-01", "2023-12"]}


def doc() -> dict:
    na = pr.not_applicable
    here = Path(__file__).parent
    inputs = [PRINTED / "MANIFEST.json", PRINTED / "printed_monthly.parquet",
              L0_NEW / "wra_cy_monthly_reports_007001" / "SOURCE.json",
              PREREG_DIR / "S2r2.deep_benchmarks_core.json",
              here / "c_test.py", here / "cy007001.py", here / "build.py", here / "s2lib.py",
              here / "transcribe_007001.py"]
    return {
        "id": PREREG_ID,
        "wp": "S2",
        "title": "Fresh test C: deep-benchmark feeds against the printed 007001 table "
                 "(pre-origin C1 and 2023 C2)",
        "kind": "test",
        "tests": [C1, C2],
        "zones": na("per-sensor series; no zones"),
        "priors": na("nothing is fitted"),
        "burn_in": na("no model run"),
        "folds": na("no cross-validation; a direct comparison of two records"),
        "fresh_units": C1["fresh_units"] + C2["fresh_units"],
        "thresholds": {
            "tolerance_mm": T.TOL_MM,
            "tolerance_rule": "|printed compression - feed monthly-mean increment| <= 2.0 mm for "
                              "EVERY testable unit (plan WP S2 threshold, read strictly as for "
                              "test A)",
            "min_r": T.MIN_R, "min_months_for_r": T.MIN_N_R,
            "correlation_rule": "Pearson r >= 0.85 for every sensor with >= 5 testable months",
            "min_full_days_per_month": T.MIN_DAYS,
            "source": "plan WP S2 thresholds (post hoc in the plan; unchanged here)",
        },
        "gate_scores": [
            {"name": "tolerance_every_unit", "weight": 0.5,
             "rule": "pass iff every testable unit is within 2.0 mm"},
            {"name": "correlation_every_sensor", "weight": 0.5,
             "rule": "pass iff every r-testable sensor has r >= 0.85"},
        ],
        "primary_model": {
            "product": "deep_benchmarks built by dataprep_staging/s2_deep_benchmarks/build.py "
                       "under the rules declared in S2r2.deep_benchmarks_core (step catalogue, "
                       "vendor-to-sensor identity, primary feed per month, flags)",
            "feed_increment": "difference of consecutive monthly means of one feed "
                              "(c_test.monthly_mean_increments): mean of the full days (daily "
                              "flags 0-1) of each month, >= 20 full days in both months; the raw "
                              "catalogued feed when it qualifies, else the vendor feed; flag 9 "
                              "when the product's month-end increment of month m or m-1 for that "
                              "feed is flag 9, 1 when a removed step lies in either month, else "
                              "0; testable flags 0-1",
            "convention_rationale": "the 007001 reports state their method: GNSS、地下水位、深層"
                                    "水準樁取每月平均值加以計算 (monthly mean values; read in "
                                    "the December 2022 report text before any comparison). "
                                    "The month-end convention of test A is reported as a "
                                    "secondary result, never substituted",
            "printed_unit": "one value per (sensor, month): the verified (flag 0) report-month "
                            "column, else the verified prior-year column of the report twelve "
                            "months later; value = compression mm, the printed number negated "
                            "or not according to each table's own sign note; months <= 2023-12 "
                            "only (2024+ overlaps the design-time units of A and B)",
            "mapping": "printed (station, depth) -> sensor: 溪州國小 300 SJES, 土庫國中 400 TKJS(3), "
                       "300 TKJS(1), 130 TKJS(2), 中科虎尾 300 NTUH(1), 70 NTUH(2), 秀潭國小 70 "
                       "STES (agency identifiers and station metadata depths)",
        },
        "secondary_reported_not_gating": [
            "the same comparison with the product's month-end increments (test A's convention)",
            "the monthly-mean comparison on the vendor feed as delivered and on the raw feed "
            "without the catalogue",
            "units the product flags 9 (e.g. TKJS(2) vendor months whose stitches conflict with "
            "the sparse raw record), compared with both feeds as delivered",
            "printed 007001 against printed integrated values for 2024-01..2025-12 (printed vs "
            "printed; no feed)",
            "the 2024-01..2025-03 overlap with the feeds, labelled design-time (units of A)",
        ],
        "inputs": [rel_to_repo(p) for p in inputs],
        "interpretation": (
            "C1 passing means that, in the pre-origin years S1 uses for the 2022 path shape, "
            "the product's feed months reproduce an agency table never consulted while the "
            "rules were set, at the plan's tolerance, under the convention the agency states. "
            "C1 failing is reported with its numbers and the failing units; the product's "
            "notes then warn pre-origin consumers, and no rule is changed in response within "
            "this version. C2 is the same check on 2023 (post-origin; a validator check, never "
            "a model gate). Neither test is part of the plan's gate (A and B); neither selects "
            "a model (U7)."),
        "design_time_exposure": (
            "Before this registration the author had: read the December 2022 report's table "
            "(values only); transcribed every report (printed values, read agreement, sign "
            "notes, header checks; no feed involved); seen the product's monthly coverage and "
            "flag counts per sensor and year and the vendor-raw stitch classification "
            "(TKJS(2) vendor months flagged 9 over 2017-2023); seen the results of the "
            "declared tests A and B on 2024-25 units. No 007001 value was compared with any "
            "feed value."),
    }


def main(dry: bool) -> int:
    d = doc()
    errs = pr.validate_prereg(d)
    if errs:
        raise SystemExit(f"{d['id']}: {errs}")
    if dry:
        print("valid:", d["id"], len(d["inputs"]), "inputs")
        return 0
    path, sha = pr.write_prereg(d)
    print(json.dumps({"id": d["id"], "path": str(path.relative_to(REPO)), "sha256": sha},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry" in sys.argv))

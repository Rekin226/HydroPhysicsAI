"""Write the round-4 declaration ``S2r4.deep_benchmarks_v2`` with the A1 tool (WP S2).

It fixes, before ``printed_v2.py``, ``label_confirm.py`` and ``build_v2.py`` are run for the
products, the rules of deep_benchmarks_printed/v2, of the pre-origin labels and of
deep_benchmarks/v2, and hashes their code. It registers no fresh test: no fresh
feed-against-printed unit remains (the feeds end in 2025-04 and every printed month up to
then has been compared by tests A, C1 or C2).

Usage: python -I prereg_r4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, rel_to_repo  # noqa: E402
from harness import prereg as pr  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
ID = "S2r4.deep_benchmarks_v2"
CODE = ["printed_v2.py", "label_confirm.py", "build_v2.py", "build.py", "s2lib.py", "c_test.py",
        "__init__.py", "prereg_r4.py", "tests/test_r4.py"]


def doc() -> dict:
    na = pr.not_applicable
    inputs = [rel_to_repo(HERE / c) for c in CODE]
    inputs += [rel_to_repo(L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet") for k in L.VENDOR_KEYS]
    inputs += [rel_to_repo(L.OPEN_DIR / f) for f in L.OPEN_FILES]
    inputs += [rel_to_repo(L.PRINTED_CSV), rel_to_repo(L.PRINTED_REPORT),
               rel_to_repo(DATAPREP / "deep_benchmarks_printed/v1/MANIFEST.json"),
               rel_to_repo(DATAPREP / "deep_benchmarks_printed/v1/printed_monthly.parquet"),
               rel_to_repo(DATAPREP / "deep_benchmarks_printed/v1/printed_cumulative.parquet"),
               rel_to_repo(DATAPREP / "deep_benchmarks/v1/MANIFEST.json"),
               rel_to_repo(DATAPREP / "deep_benchmarks_printed/v2/MANIFEST.json"),
               rel_to_repo(DATAPREP / "deep_benchmarks_printed/v2/printed_monthly.parquet")]
    return {
        "id": ID, "wp": "S2", "kind": "declaration",
        "title": "Deep benchmarks round 4: deep_benchmarks_printed/v2 (one primary printing "
                 "per sensor-month, integrated values under a manifest), pre-origin labels of "
                 "the vendor series, and deep_benchmarks/v2 whose pre-origin rows come from a "
                 "pre-origin run; gate A and B unchanged; sd labelled uncalibrated",
        "tests": na("no fresh test: every printed month that overlaps a feed (2019-05.."
                    "2025-04) was already compared by test A (integrated reports, design-time "
                    "units), C1 or C2 (007001 table, registered S2r2.c_fresh_cy007001 and "
                    "scored on v1). A, B and C are recomputed on v2 as after-the-fact "
                    "statistics (looks logged)."),
        "zones": na("no spatial zones; per-sensor series"),
        "priors": na("no priors"),
        "burn_in": na("no model run"),
        "folds": na("no cross-validation"),
        "fresh_units": na("none (see tests)"),
        "thresholds": {
            "A_printed_vs_feed": "unchanged from S2r2.deep_benchmarks_core: |printed - feed| "
                                 "<= 2.0 mm for every unit (flags 0-1), integrated reports "
                                 "2024-01..2025-04, month-end convention",
            "B_mapping_correlation": "unchanged: Pearson r >= 0.85 for every printed pile with "
                                     ">= 5 overlapping months (flags 0-2). Reported in full: "
                                     "every mapped sensor is listed with why it is or is not "
                                     "testable",
            "printed_v2_gate": "share of printed 007001 monthly values verified by two reads "
                               ">= 0.90 (v1 rule), no verified row whose month headers "
                               "disagree with its caption (v1 rule), and share of non-null "
                               "integrated-report monthly values two-read verified >= 0.90",
            "preorigin_labels": "rule text of label_confirm.py (docstring 'Rule'): (1) identity "
                                "on pre-origin hours (share of informative common hours with "
                                "an unchanged offset or sum >= 0.98 over >= 48 hours); (2) "
                                "else the agency station code equal to the vendor key (SJES, "
                                "STES, WNES, DLIO); (3) else the printed 007001 table up to "
                                "2022-12 (deep_benchmarks_printed/v2 primary printings, flag "
                                "0, read as design_time_evidence through the 2022-12 slice): "
                                "NTUH_281_29/NTUH_282_29 take the assignment under which each "
                                "correlates better (>= 5 months) with its own depth than with "
                                "the other, else ambiguous; TKJS_283_29 takes TKJS(3) when its "
                                "r with the printed 400 m column (>= 5 months) is >= the "
                                "smaller r of the identity-confirmed TKJS_281_29/TKJS_282_29 "
                                "against their own depths on the same months, else defective. "
                                "Vendor statistic: differences of consecutive monthly means "
                                "(days >= 12 valid hours, or every sampled day for a "
                                "once-a-day series; months >= 20 days); no step catalogue",
        },
        "gate_scores": {"deep_benchmarks_v2": "A and B both pass (logical AND), as v1",
                        "deep_benchmarks_printed_v2": "the three printed_v2_gate conditions "
                                                      "(logical AND)"},
        "primary_model": {
            "deep_benchmarks_v2": [
                "pre-origin run: vendor samples and raw agency hours cut before "
                "2023-01-01T00:00; vendor labels from label_confirm (status confirmed -> the "
                "label; ambiguous -> the plan's nominal label with every pre-origin vendor row "
                "of the sensor flag 9 (monthly flag and vendor_flag; monthly-mean increments "
                "computed with those flags, then the daily vendor rows set to 9); defective -> "
                "flag 9 rows as v1); no printed integrated value (they start 2024-01); "
                "cumulative series solved on pre-origin increments anchored at 0 at the end "
                "of 2022-12",
                "full run: v1's computation and labels; must reproduce v1's tables (checked)",
                "assembly: rows dated <= 2022-12 (daily date, month, a step's t_after month) "
                "from the pre-origin run, later rows from the full run; column 'variant'; "
                "monthly-mean increments within each run; tables without a month column hold "
                "both runs",
                "the step envelope Q(d), floor and sigma_day as v1 (pre-origin, identical in "
                "both runs; checked)",
                "merged_monthly carries the printed 007001 value of "
                "deep_benchmarks_printed/v2 (primary printing) beside the merge, not as "
                "primary",
            ],
            "deep_benchmarks_printed_v2": [
                "printed_monthly: one row per sensor and data month; primary = the verified "
                "printing of the earliest vintage (report caption month), else the earliest "
                "printing by flag (9 before 8); the other verified printing in reprint_*",
                "printed_cumulative: one row per sensor and printed period (start, end from "
                "the ROC label); month = the period's last month",
                "integrated_monthly: the 007003 values of results/twin/report_extract/"
                "deep_piles/deep_piles.csv (printed numbers and transcription checks only; the "
                "feed-derived columns left out), compression mm, verified = two reads agree",
                "time slicing on the data month; params_fit_window_end 2019-04 (nothing "
                "fitted)",
            ],
        },
        "uncertainty": "plan 4.7: every published sd is labelled 'uncalibrated' (column "
                       "sd_status, README, manifest). Checks, reported and never used to "
                       "rescale: coverage of the nominal 95 % interval of the month-end "
                       "increments' sd (with the printed rounding sd) on the test-A "
                       "differences; coverage of the printed rounding sd on the differences "
                       "between two printings of the same month (007001).",
        "anti_leakage": "No observation dated 2023-01 or later enters a parameter, label or "
                        "flag of a row dated up to 2022-12 in deep_benchmarks/v2 (U3, U7). "
                        "The printed products fit nothing. A re-printed value is the agency's "
                        "value for its data month; its vintage is recorded. The gate selects "
                        "no model.",
        "design_time_exposure": [
            "All v1 results: A 58/61 (fails), B min r 0.972, C1 170/207 (fails; by year 29/32, "
            "49/61, 39/61, 53/53), C2 46/46; per-sensor C1 r with v1 labels NTUH(1) 0.815, "
            "NTUH(2) 0.750, SJES 0.829, STES 0.794, TKJS(1) 0.864, TKJS(2) 0.789; round-3 "
            "diagnostics.",
            "The round-3 verifier's findings: 17 pre-origin TKJS(2) months flag 9 through the "
            "2021-08-19..2023-05-11 stitch; v1 labels rest on 2023-25 hours; TKJS_283_29 is a "
            "daily series (verifier: daily-change r -0.10 with raw TKJS(3) over 291 days; r "
            "-0.76 against printed 400 m increments 2021-22, 3 of 20 within 2 mm); sd "
            "uncalibrated; printed months double-counted.",
            "A scratch run of printed_v2.py by the interrupted repair attempt (about "
            "2026-10-12T11:12Z): 479 sensor-months, 467 verified, 21 pre-origin sensor-months "
            "known only from a 2023 re-printing, rounding-sd coverage 0.80 on 245 re-printed "
            "pairs (9 differ by > 2 mm).",
            "deep_benchmarks_printed/v2 was built (2026-10-12T15:42Z) before this file, with "
            "the rules of printed_v2.py's docstring (no score enters it; its gate concerns "
            "transcription shares known from v1): 479 sensor-months, 467 verified, gate "
            "passed; it is an input here.",
            "Development runs of build_v2.py with v1's labels in place of the pre-origin "
            "labels (looks logged 15:42-15:43Z): the full run reproduced v1's tables; the "
            "pre-origin monthly rows differed from v1 only in the 17 TKJS(2) flags (9 -> 0); "
            "A 58/61 and B min r 0.972 as v1; test C's statistic on those rows C1 184/221 "
            "(TKJS(2) r 0.908, the other sensors as v1), C2 46/46; month-end sd coverage on "
            "the test-A units 0.84 at nominal 0.95; TKJS_283_29 daily-change r -0.096 with "
            "raw TKJS(3) over 289 day pairs. A second development run (--no-log) recomputed "
            "the same numbers after edits to the TKJS_283_29 wording.",
            "The vendor station metadata (ls-wra-dbm-obs__meta.parquet, non-personal "
            "columns) gives file names and column orders but no vendor key, and the column "
            "order does not map 281/282 consistently for TKJS and NTUH: not used for labels.",
            "Not seen before this file: the correlations of the uncatalogued vendor series "
            "with the printed table (label_confirm), and every v2 statistic.",
        ],
        "inputs": inputs,
        "interpretation": (
            "deep_benchmarks/v2 is a corrected version of v1, not a new validation: its gate "
            "uses v1's tests and units and is expected to fail as v1's did; it is published "
            "with gate.passed as computed. Its value is a pre-origin time slice that is truly "
            "pre-origin, sd labelled, correct justifications and a full B scope. "
            "deep_benchmarks_printed/v2 changes no printed number; it removes double "
            "counting and puts the integrated values under the guard. A label found "
            "ambiguous makes the sensor's pre-origin rows unusable for pre-origin targets "
            "(flag 9); it does not change post-origin rows."),
        "supersedes_for_products": ["results/dataprep/deep_benchmarks/v1",
                                    "results/dataprep/deep_benchmarks_printed/v1"],
    }


def main() -> int:
    path, sha = pr.write_prereg(doc())
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

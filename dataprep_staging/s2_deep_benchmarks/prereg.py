"""Write the S2 acceptance pre-registration before the product is built and scored.

The WP A1 pre-registration tool did not exist on 2026-10-12, so this script writes the JSON in
the layout the other Phase-0 WPs used (``results/dataprep/preregistration/<id>.json``),
refusing to overwrite. It fixes the step-catalogue rule, the monthly convention, the two
acceptance tests of plan WP S2 and the gate before any printed-against-feed number of the
product exists. Design-time exposure is declared in the file.

Usage: python -I prereg.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

PREREG_ID = "S2.deep_benchmark_acceptance.v1"


def main() -> int:
    inputs = [L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet" for k in L.VENDOR_KEYS]
    inputs += [L.OPEN_DIR / f for f in L.OPEN_FILES]
    inputs += [L.PRINTED_CSV, L.PRINTED_REPORT, L.SITES_CSV]
    rec = {
        "id": PREREG_ID,
        "wp": "S2",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP S2 (acceptance)",
        "question": "After a step catalogue built only from the feeds and from pre-origin "
                    "statistics, do the deep-benchmark feeds reproduce the printed monthly "
                    "values of the WRA monthly reports (2024-01..2025-04 overlap), and does "
                    "every printed pile map onto one feed sensor?",
        "inputs": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in inputs],
        "code_at_registration": [{"path": rel_to_repo(Path(L.__file__)),
                                  "sha256": sha256_file(Path(L.__file__))}],
        "step_catalogue_rule": {
            "feeds": "every in-domain feed series: raw agency feed (open data 58339) and vendor "
                     "feed; vendor TKJS_283_29 is excluded if it has no identity with any raw "
                     "sensor (then flagged 9, defective)",
            "sign": "raw hours inside a sign-inverted segment of the vendor identity "
                    "(vendor = -raw + c over >= 48 informative hours) are negated before "
                    "detection; the inversion runs to the next same-sign segment or the record "
                    "end",
            "discontinuity": f"consecutive valid hours more than {L.GAP_H} h apart (outage), or "
                             "a consecutive pair whose change exceeds the threshold for its "
                             f"duration (jump); candidates closer than {L.MERGE_H} h merge",
            "statistic": "plain change across the discontinuity (glob) or its residual against "
                         "the local rate (mean least-squares slope of up to "
                         f"{L.SIDE_H} h on each side, >= {L.MIN_SIDE_PTS} valid hours, bounded "
                         "by the neighbouring discontinuities) times the duration (rate); at "
                         "each duration the statistic whose pre-origin envelope is smaller",
            "envelope": f"Q(d) = quantile {L.Q_LEVEL} of |statistic| on artificial gaps of "
                        f"durations {list(L.DURS_H)} h placed in continuous runs (no gap > "
                        f"{L.GAP_H} h over the span +- {L.SIDE_H} h) of the vendor hourly feeds "
                        "TKJS_281_29, TKJS_282_29, NTUH_281_29, NTUH_282_29, SJES, STES, WNES "
                        "up to 2022-12-31 23:00 (3000 draws per sensor and duration, seed "
                        "2026); log-log interpolation in d, sqrt growth beyond 1440 h",
            "threshold": f"max(Q(d), {L.FLOOR_MM} mm) (the 1 mm floor is a priori)",
            "classes": "step (above threshold: removed from every later value, flag 1 after "
                       "it, correction sd = robust sd of the statistic at d); spike (flagged "
                       "jumps whose span is not a step but whose interior deviates beyond the "
                       "threshold: interior hours removed, flag 9); outage (change kept as real "
                       "movement); noise (no action)",
            "vendor": "the vendor feed is kept as delivered; a vendor increment is flagged 9 "
                      "when its interval holds a vendor stitch (identity offset change) that "
                      "the raw catalogue classifies as real movement, or a raw step the "
                      "vendor did not stitch, or a step detected in the vendor series itself",
        },
        "monthly_convention": {
            "level": "daily mean of the valid hours of the last day of the month (flag 0); a "
                     f"partial last day (< {L.DAY_MIN_HOURS} h) or the latest day with a value "
                     f"up to {L.BRIDGE_DAYS} days earlier gives flag 2",
            "increment": "level(m) - level(m-1), cm; flag 8 if an endpoint is missing, 9 if a "
                         "suspect event lies inside, 2 if an endpoint is bridged or partial, 1 "
                         "if a removed step lies inside, else 0",
            "sd": "sqrt(2) x sigma_day (robust day-to-day noise about a centred 7-day linear "
                  "fit, 2022-12 or earlier, per sensor; pooled median for sensors without "
                  "pre-origin data) plus the correction sd of each removed step and of each "
                  "bridged endpoint, in quadrature",
            "primary_feed": "raw-catalogued increment where its flag is 0-2, else the vendor "
                            "increment",
        },
        "mapping": {
            "printed_to_sensor": {f"{k[0]} {k[1]} m": v for k, v in L.PRINTED_TO_SENSOR.items()},
            "basis": "agency identifiers, names and depths (open-data feed and station "
                     "metadata); vendor series to sensor by identity (share of informative "
                     "hours with an unchanged offset or sum >= 0.98)",
        },
        "primary_tests": {
            "A_printed_vs_feed": {
                "units": "every (printed pile, depth) with a mapped sensor and every month with "
                         "a verified, non-null printed monthly value and a primary-feed "
                         "increment of flag 0 or 1",
                "metric": "|printed - feed| in mm",
                "criterion": "<= 2.0 mm for every unit (plan WP S2 threshold, read strictly)",
            },
            "B_mapping_correlation": {
                "units": "every (printed pile, depth) with >= 5 months where the printed value "
                         "and the mapped sensor's primary-feed increment (flags 0-2) exist",
                "metric": "Pearson r of printed against feed monthly values",
                "criterion": ">= 0.85 for every unit (plan WP S2 threshold)",
            },
            "gate": "A and B both pass",
        },
        "secondary_reported_not_gating": [
            "A with flag-2 increments included; A on the vendor feed as delivered; A on the raw "
            "feed without the catalogue; A restricted to the four piles on the model grid",
            "B for every sensor (best match by r and by median absolute difference)",
            "recall of injected steps (2, 5, 10, 20 mm) at 1-168 h on pre-origin data and the "
            "observed false-positive share",
            "the plan's expected steps (2024-02-29/03-01 at every pile, the 溪州 reset in March "
            "2025, 灣內's sign) found or not",
            "agreement of the raw catalogue with the vendor's stitches; vendor stitches that "
            "removed movement the catalogue keeps",
            "cumulative compression per pile from printed (primary) and feed increments, with "
            "printed trailing-twelve-month values as extra constraints",
        ],
        "design_time_exposure": (
            "Before this file was written the author saw (i) the pilot mapping table "
            "(_pilots/subsidence_gap_sources/out/dbm_mapping.csv: raw-feed correlations 0.877-"
            "0.994 without a step catalogue), (ii) the report_extract feed check (44/59 within "
            "0.15 cm), and (iii) during exploration on 2026-10-12, month-end differences of both "
            "feeds against the printed values for 2024-01..2025-04 and the raw-against-vendor "
            "changes across vendor gaps (for example 秀潭 2024-04 and 2024-07, where the printed "
            "values agree with the raw feed and not with the vendor). The step rule's numbers "
            "come only from pre-origin statistics; its per-duration choice of statistic was "
            "fixed from the pre-origin envelope (rate-adjusted residual tighter up to about 1 "
            "day, plain change tighter beyond about 2 days), not from agreement with the "
            "printed values. Both thresholds are the plan's, set after the pilot (post hoc)."),
        "anti_leakage": (
            "No observation dated 2023-01 or later enters Q(d), the floor, sigma_day or any "
            "other numeric parameter (U7). The gate is a data-consistency check of the "
            "validator itself and necessarily uses its 2024-25 values; it selects no model. The "
            "product forbids selection, model_design, prior, prior_width_input, "
            "initial_conditions, idw_field and forcing uses; pre-origin uses (datum_fit, "
            "calibration_target, baseline, design_time_evidence) are reachable only through a "
            "time slice ending by 2022-12 (time_slicing column 'month', params_fit_window_end "
            "2022-12). Printed values are never filled; feed fills of printed gaps carry the "
            "feed's own flags."),
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "written_by": "dataprep_staging/s2_deep_benchmarks/prereg.py (A1 tool not available "
                      "yet)",
    }
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    p = PREREG_DIR / f"{PREREG_ID}.json"
    with open(p, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(p, sha256_file(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

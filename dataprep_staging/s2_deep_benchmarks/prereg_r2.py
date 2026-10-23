"""WP S2 round 2: record the design-time looks and register the core rules with the A1 tool.

Round 1 wrote ``S2.deep_benchmark_acceptance.v1.json`` with its own script before the A1
pre-registration tool existed; that file is not in ``INDEX.jsonl`` and the tool cannot load
it. Round 2 therefore

1. records, as a back-dated design-time look (``purpose="design_time_seed"``), the three
   occasions on which printed deep-pile values of the integrated monthly reports were compared
   with the feed for 2024-01..2025-04 before any S2 rule was fixed: the report_extract feed
   check (2026-10-11), the subsidence pilot's mapping table (2026-10-12 00:40 +0800) and round
   1's exploration (2026-10-12 10:27-10:33 +0800). Those pile-months can never be fresh units;
2. registers a **declaration** (kind ``declaration``) that carries the v1 rules verbatim, the
   plan's acceptance tests A and B with their thresholds, the gate, the secondary diagnostics
   and the three code corrections made in round 2 before anything was scored. A and B are
   data-consistency checks of the validator on design-time units, so they are declared, not
   registered as fresh tests. A fresh test, if one is possible, is registered separately
   before it is scored.

Usage: python -I prereg_r2.py [--dry]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PREREG_DIR, REPO, rel_to_repo  # noqa: E402
from harness import prereg as pr  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

DECL_ID = "S2r2.deep_benchmarks_core"
V1 = PREREG_DIR / "S2.deep_benchmark_acceptance.v1.json"
SENSORS_PRINTED = ["SJES", "TKJS(3)", "TKJS(1)", "STES", "NTUH(1)", "WNES", "DLIO"]
SEED_UNITS = [f"dbm_integrated_printed:{s}" for s in SENSORS_PRINTED]
SEED_SCOPE = "deep_piles_integrated_printed_vs_feed:2024-01..2025-04"
SEED_TIME = "2026-10-12T02:27:00.000000Z"
SEED_NOTE = ("recorded by S2 round 2. Printed integrated-report deep-pile values were compared "
             "with the feeds before any S2 rule was fixed: (i) twin.report_extract feed check, "
             "2026-10-11 16:31 +0800 (results/twin/report_extract/deep_piles/report.json: 44 of "
             "59 within 0.15 cm, uncatalogued raw feed); (ii) subsidence pilot dbm_daily.py, "
             "2026-10-12 00:40 +0800 (_pilots/subsidence_gap_sources/out/dbm_mapping.csv, r "
             "0.877-0.994, vendor feed); (iii) S2 round-1 exploration, 2026-10-12 10:27-10:33 "
             "+0800 (month-end differences of both feeds against the printed values, disclosed in "
             "S2.deep_benchmark_acceptance.v1.json). Seeded at the time of (iii), the latest.")


def inputs() -> list[str]:
    files = [L.VENDOR_DIR / f"ls-wra-dbm-obs__{k}.parquet" for k in L.VENDOR_KEYS]
    files += [L.OPEN_DIR / f for f in L.OPEN_FILES]
    files += [L.OPEN_DIR / "SOURCE.json", L.PRINTED_CSV, L.PRINTED_REPORT, L.SITES_CSV,
              L.GRID_NPZ, V1, Path(L.__file__), Path(__file__).with_name("build.py")]
    return [rel_to_repo(p) for p in files]


def declaration() -> dict:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    na = pr.not_applicable
    return {
        "id": DECL_ID,
        "wp": "S2",
        "title": "Deep benchmarks (深層樁) as a first-class series: core rules, acceptance tests "
                 "A and B and gate, carried over from the round-1 file with three code "
                 "corrections",
        "kind": "declaration",
        "tests": na("the plan's tests A (printed vs feed <= 2 mm) and B (mapping r >= 0.85) are "
                    "scored on printed pile-months 2024-01..2025-04 that were compared with the "
                    "feeds three times before any rule was fixed (seeded look "
                    f"{SEED_SCOPE}); they are data-consistency checks of the validator and are "
                    "declared here with their thresholds, before the product is built, rather "
                    "than registered as fresh tests. A fresh test is registered separately "
                    "before it is scored."),
        "zones": na("no spatial zones; per-sensor series"),
        "priors": na("no priors; nothing is fitted except the pre-origin step envelope and the "
                     "daily noise, both from data <= 2022-12"),
        "burn_in": na("no model run"),
        "folds": na("no cross-validation; the envelope is estimated on artificial gaps in "
                    "pre-origin vendor data"),
        "fresh_units": na("none for A and B (design-time units); see the separate fresh-test "
                          "registration"),
        "thresholds": {
            "A_printed_vs_feed": v1["primary_tests"]["A_printed_vs_feed"],
            "B_mapping_correlation": v1["primary_tests"]["B_mapping_correlation"],
            "step_threshold": v1["step_catalogue_rule"]["threshold"],
            "identity": "a vendor series is the copy of a raw sensor when the share of "
                        "informative common hours (|vendor hourly change| > 0.015 mm) with an "
                        "unchanged offset OR an unchanged sum is >= 0.98 (>= 48 common hours)",
            "source": "plan WP S2 (2 mm, 0.85; post hoc, set after the pilot); identical to "
                      "S2.deep_benchmark_acceptance.v1.json",
        },
        "gate_scores": {"rule": v1["primary_tests"]["gate"],
                        "weights": "not applicable: both tests must pass (logical AND)"},
        "primary_model": {
            "step_catalogue_rule": v1["step_catalogue_rule"],
            "monthly_convention": v1["monthly_convention"],
            "mapping": v1["mapping"],
            "merge": "printed monthly values of the integrated reports (verified transcription) "
                     "are primary where they exist; the feed (primary feed per month) fills "
                     "other months with its own flags; printed values are never filled",
            "cumulative": "weighted least squares on merged increments (flags 0-2) plus printed "
                          "trailing-twelve-month values of the same sensor, referenced to the "
                          "end of 2022-12 where connected",
            "secondary_reported_not_gating": v1["secondary_reported_not_gating"],
        },
        "inputs": inputs(),
        "interpretation": (
            "A passed gate (A and B) means that, after a step catalogue whose numbers come only "
            "from pre-origin statistics, the feed reproduces the agency's printed monthly "
            "values at every testable pile-month and every printed pile maps onto one feed "
            "sensor. Because the units were examined before the rules were fixed, a pass is "
            "design-time consistency, not fresh validation; a failure is reported with its "
            "numbers and the product is still written with gate.passed = false (the loader "
            "guard then refuses validator, scoring and pre-origin uses). The gate selects no "
            "model and feeds no threshold of any model (U7)."),
        "changes_from_v1": [
            "pandas 2.3.3 mis-aligns two regular microsecond DatetimeIndexes (union returns a "
            "few wrong timestamps; reproduced on synthetic ranges); every series now gets a ns "
            "index before resampling or alignment (s2lib.as_ns). Robustness only: the round-1 "
            "pairs mixed us and ns and aligned correctly.",
            "identity rule implemented as written (union of 'unchanged offset' and 'unchanged "
            "sum' hours); round-1 code used the larger of the two shares, which would have "
            "marked the sign-inverting 灣內 vendor series defective (shares 0.64 + 0.36 on "
            "2023-25 common hours; no printed value involved).",
            "sign segments implemented as written ('the inversion runs to the next same-sign "
            "segment'); round-1 code passed only the inverted segments, so an inversion could "
            "never end. No effect on the present data (灣內 has one inversion, 2024-08-08, "
            "running to the record end).",
            "build: the last product month is the last verified printed month (was hard-coded "
            "2026-01); time_slicing declares encoding 'iso' and is checked with "
            "guard.time_slicing_errors before the manifest is written (A0fix); lint fixes.",
        ],
        "design_time_exposure": {
            "round_1": v1["design_time_exposure"],
            "round_2_before_registration": (
                "The round-2 author read the round-1 code, the A0/A0fix reports and the pilot "
                "mapping table (r 0.877-0.994), and computed: (a) vendor-to-raw identity shares "
                "and sign segments on 2023-25 common hours (feeds only); (b) the pre-origin "
                "envelope Q(d) and injected-step recall (<= 2022-12); (c) while checking the "
                "WRA site for newer reports, the Table 1 printed values of one Changhua-Yunlin "
                "monthly data report (007001 series, December 2022: 7 piles, months 2021-12 "
                "and 2022-12), with no feed comparison. No printed-against-feed number was "
                "computed by round 2 before this registration."),
        },
        "anti_leakage": v1["anti_leakage"],
        "supersedes_for_scoring": rel_to_repo(V1),
    }


def main(dry: bool) -> int:
    d = declaration()
    errs = pr.validate_prereg(d)
    if errs:
        raise SystemExit(f"{d['id']}: {errs}")
    if dry:
        print("valid:", d["id"], len(d["inputs"]), "inputs")
        return 0
    looks = pr.read_chained(PREREG_DIR / "looks.jsonl")
    if not any(lk.get("wp") == "S2" and lk.get("data_scope") == SEED_SCOPE for lk in looks):
        pr.record_look(SEED_UNITS, SEED_SCOPE, "S2", "design_time_seed", seeded_time=SEED_TIME,
                       note=SEED_NOTE)
    path, sha = pr.write_prereg(d)
    print(json.dumps({"id": d["id"], "path": str(path.relative_to(REPO)), "sha256": sha},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry" in sys.argv))

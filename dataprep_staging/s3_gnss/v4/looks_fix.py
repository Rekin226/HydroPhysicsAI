"""WP S3 fix round of the S3r repair (2026-10-18): record looks in the A1 look log with
``harness.prereg3`` BEFORE the computation they describe.

    python -I -B dataprep_staging/s3_gnss/v4/looks_fix.py KEY [KEY ...]

Each KEY names one look below; entries are added to this file before the look is recorded
(the file is not a registered input). A look already recorded with the same scope and note
is skipped. New units carry members (prereg3 requires them for units outside the registries).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # dataprep_staging
from harness import prereg3  # noqa: E402

V3 = "S3.gnss_clean_acceptance.v3"
SYN = ("synthetic step injections on GNSS height series; no registered well, leveling "
       "benchmark, fold test set or head-data cell")


def table_members(unit: str, window: str, what: str) -> dict:
    return {unit: {"ids": ["gnss:all_wra_ngl"], "window": window, "description": what}}


LOOKS = {
    "fix_diag_cm_composition": dict(
        units=["s3.product_tables.v3_full_base", "s3.product_tables.v3_preorigin_base"],
        data_scope="gnss_wra_series:2020-01..2025-03", purpose="diagnose",
        members={**table_members("s3.product_tables.v3_full_base", "2020-01..2025-03",
                                 "v3 full-variant WRA family run (no injection): detection "
                                 "common-mode composition, neighbour availability, steps "
                                 "and flags"),
                 **table_members("s3.product_tables.v3_preorigin_base", "2020-01..2022-12",
                                 "v3 pre-origin WRA family run (no injection): the same "
                                 "composition diagnostics")},
        note="S3r fix round, before any method change: the verifier's major issue (R6 "
             "5-nearest-station WRA detection common mode jumps when neighbours return after "
             "the 2022-23 gap). Re-runs the v3 WRA family (full and pre-origin, no "
             "injection) with the v3 code and tune_v3c, and tabulates per station and day "
             "which of the 5 neighbours report, the common-mode jump at every change of "
             "composition split into a composition part and a constant-subset part, and the "
             "common-mode-only steps that coincide with composition changes; reads the v3 "
             "full product steps/flags. No injection, no A1/A2/A3 score"),
    "fix_diag_cm_composition_ngl": dict(
        units=["s3.product_tables.v3_full_base_ngl", "s3.product_tables.v3_preorigin_base_ngl"],
        data_scope="gnss_ngl_series:2007-01..2026-09", purpose="diagnose",
        members={**table_members("s3.product_tables.v3_full_base_ngl", "2007-01..2026-09",
                                 "v3 full-variant NGL family run (no injection): detection "
                                 "common-mode composition (off-fan reference median)"),
                 **table_members("s3.product_tables.v3_preorigin_base_ngl", "2007-01..2022-12",
                                 "v3 pre-origin NGL family run (no injection): the same")},
        note="S3r fix round, before any method change: does the NGL detection common mode "
             "(leave-one-out median of the off-fan reference stations, unchanged since v1) "
             "have the same composition defect as the WRA 5-nearest median? Re-runs the v3 "
             "NGL family (full and pre-origin, no injection) and tabulates the common-mode "
             "jump at every reference-station gap or record start/end, split into "
             "composition and constant-subset parts, and the common-mode-only NGL steps near "
             "such events. No injection, no A1/A2/A3 score"),
    "fix_diag_product_cm": dict(
        units=["s3.product_tables.v3_full_base"],
        data_scope="gnss_wra_series:2022-11..2023-03", purpose="diagnose",
        note="S3r fix round, before any method change: composition part of the WRA network "
             "median (product common modes wra_onfan_fullband and wra_hf of gnss_clean/full/"
             "v3) at the 2023-01-07 and 2023-02-08 outage edges, from the v3 full product "
             "daily table (members, flags 0/1/3); no injection, no A1/A2/A3 score"),
    "fix_smoke_v4": dict(
        units=["s3.product_tables.v4_smoke"],
        data_scope="gnss_full_records:2007-01..2026-09", purpose="smoke_test",
        members=table_members("s3.product_tables.v4_smoke", "2007-01..2026-09",
                              "v4 (R7) family runs of both variants, no injection: "
                              "composition check, identity with v3 on full-composition "
                              "days, steps/flags against v3, NGL identity"),
        note="S3r fix round, smoke check declared in S3.repair.md 7.4 item 1 (smoke_v4.py): "
             "v4 WRA family (R7) on both variants without injection; composition parts of "
             "every neighbour event under R7; maximum |v4 - v3 median| on days with all 5 "
             "neighbours (expected exactly 0); v4 tables of both variants against v3 "
             "(steps added/removed, flag-0/1 days in 2023-25, NGL tables identical apart "
             "from product_version); run times. No injection, no A1/A2/A3 score"),
    "fix_smoke_v4b": dict(
        units=["s3.product_tables.v4_smoke"],
        data_scope="gnss_full_records:2007-01..2026-09", purpose="smoke_test",
        note="S3r fix round, smoke check of R7' (pairwise offsets; S3.repair.md 7.5) with "
             "smoke_v4.py, same content as fix_smoke_v4: composition parts under R7', "
             "identity with v3 on full-composition days, v4 tables against v3 (steps, flag-0/1 "
             "days 2023-25, NGL identity), station notes for the stations named in 7.2. No "
             "injection, no A1/A2/A3 score"),
    "fix_matched_sets": dict(
        units=["s3.twin_score.gnss_rates.wra.2023-01..2025-03", "s3.a3.matched_sets"],
        data_scope="gnss_full_records:2020-01..2025-03", purpose="after_the_fact",
        prereg_id=V3,
        members={"s3.a3.matched_sets": {
            "ids": ["gnss:wra_ring_pairs", "ring_or_pile:co-located_rings"],
            "window": "2020-01..2025-03",
            "description": "A3 excess-rate change at co-located rings, v2/v3/v4 full "
                           "variants on matched site sets (report only, no threshold)"}},
        note="S3r fix round (matched_sets.py), after the v4 full validation run: A3 (both "
             "statistics) and the after-the-fact twin scoring (raw_cleaned, wra_hf, "
             "ngl_offfan) of the gnss_clean full v2, v3 and v4 on each version's own set, on "
             "the stations common to all three and to v2 and v4; flag-0/1 days 2023-25 per "
             "station; reported, never a gate (U7)"),
    "fix_trace_repro": dict(
        units=["s3.product_tables.v4_smoke"],
        data_scope="gnss_wra_series:2022-11..2024-07", purpose="smoke_test",
        note="S3r fix round (adhoc/trace_repro.py): reproduces into evidence files the "
             "ad-hoc traces quoted in S3.repair.md 7.5/7.6 (HLES and KTES neighbour residuals "
             "and detection common modes: plain median, withdrawn R7 re-implemented as it ran, "
             "and R7'; 2022-11..2023-03 and 2024-02..2024-07); no injection, no score"),
}


def main(keys: list[str]) -> int:
    existing = prereg3.read_chained(prereg3.PREREG_DIR / "looks.jsonl")
    for key in keys:
        lk = LOOKS[key]
        dup = [e for e in existing if e["wp"] == "S3" and e["data_scope"] == lk["data_scope"]
               and e["note"] == lk["note"]]
        if dup:
            print("already recorded:", key)
            continue
        rec = prereg3.record_look(lk["units"], lk["data_scope"], "S3", lk["purpose"],
                                  members=lk.get("members"), prereg_id=lk.get("prereg_id"),
                                  test_id=lk.get("test_id"), note=lk["note"])
        print("recorded", key, rec["time"], rec["purpose"], rec["data_scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

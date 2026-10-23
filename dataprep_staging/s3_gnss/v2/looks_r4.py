"""WP S3 repair round of 2026-10-13: record looks in the A1 look log with ``harness.prereg3``
(the tool A1 repair round 2 prescribes; it requires members for units outside the registries).

    python -I dataprep_staging/s3_gnss/v2/looks_r4.py KEY [KEY ...]

Each KEY names one look below. A look already recorded with the same scope and note is
skipped. Looks are recorded BEFORE the computation they describe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # dataprep_staging

sys.path.insert(0, str(Path(__file__).resolve().parent))   # v2 modules
import units_v2 as U  # noqa: E402
from harness import prereg3  # noqa: E402

V1 = "S3.gnss_clean_acceptance.v1"
V2 = "S3.gnss_clean_acceptance.v2"
A2_UNITS = U.A2_UNITS
a2_members = U.a2_members
seed_members = U.seed_members


SEED_R3 = ["s3.injection.seed101.wra.preorigin", "s3.injection.seed101.ngl.preorigin",
           "s3.injection.seed202.wra.preorigin", "s3.injection.seed202.ngl.preorigin"]

LOOKS = {
    "tune_r3_rerun": dict(
        units=SEED_R3, data_scope="gnss_injection:preorigin_background:tune_r3",
        purpose="design_time_seed", members=seed_members(SEED_R3),
        note="S3 repair round 2026-10-13: tune round r3 rerun to completion from byte-identical "
             "v2 code (the 2026-10-12 night run was cut off after 12 of 20 configurations); "
             "seeds 101/202, pre-origin series only; output results/dataprep/reports/"
             "S3_evidence/repair_2026-10-13/tune_r3/"),
    "smoke_build_full": dict(
        units=["s3.product_tables.v2_smoke"], data_scope="gnss_full_records:2007-01..2026-09",
        purpose="smoke_test",
        members={"s3.product_tables.v2_smoke": {
            "ids": ["gnss:all_wra_ngl_ncku"], "window": "2007-01..2026-09",
            "description": "v2 product tables (steps, flags), crash test only"}},
        note="S3 repair round 2026-10-13: crash test of the v2 table build (full variant, "
             "round-r2 detector values, 21-day wra_hf), before the v2 pre-registration; prints "
             "table row counts and the WRA method counts only; no A1/A2/A3 or twin score"),
    "a2_v2_pairs_diag": dict(
        units=A2_UNITS, data_scope="gnss_wra+rings+piles:2020-01..2021-12",
        purpose="diagnose", prereg_id=V2, members=a2_members("2020-01..2021-12"),
        note="S3 repair round 2026-10-13, after the v2 gate run: why the evaluable A2 pairs "
             "changed from v1 (GFES now evaluable, TKJS and YCES not): monthly flag-0/1 "
             "coverage of January/December 2020-21, steps and flags of GFES, TKJS, YCES, "
             "KTES, HLES, STES in gnss_clean/preorigin/v1 and v2; no new criterion"),
    "product_checks_v2": dict(
        units=["s3.product_tables.v1_v2"], data_scope="gnss_full_records:2007-01..2026-09",
        purpose="diagnose",
        members={"s3.product_tables.v1_v2": {
            "ids": ["gnss:all_wra_ngl_ncku"], "window": "2007-01..2026-09",
            "description": "published gnss_clean v1/v2 tables: station sets, steps, flags, "
                           "manifests"}},
        note="S3 repair round 2026-10-13, after the v2 gate run and build: repair_checks.py "
             "(v1 against v2 station sets, re-test drops, flag shares, FRES/ANES flags, "
             "manifest records), report_v2.py tables (A1/A2/A3, scatter, twin scores as "
             "computed by the registered runs) and the loader-guard probes; no new score"),
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

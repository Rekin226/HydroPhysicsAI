"""WP S3 repair round 3 (S3r, 2026-10-18): record looks in the A1 look log with
``harness.prereg3`` BEFORE the computation they describe.

    python -I dataprep_staging/s3_gnss/v3/looks_r5.py KEY [KEY ...]

Each KEY names one look below. A look already recorded with the same scope and note is
skipped. Units of the failed tests keep the ids the v1/v2 rounds gave them; new units carry
members (prereg3 requires them for units outside the registries).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # dataprep_staging
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))   # v2 unit ids
import units_v2 as U  # noqa: E402
from harness import prereg3  # noqa: E402

V1 = "S3.gnss_clean_acceptance.v1"
V2 = "S3.gnss_clean_acceptance.v2"
SYN = U.SYN

SPENT_INJECTION = (
    [f"s3.injection.seed{s}.{f}.preorigin" for s in (101, 202, 303, 404) for f in ("wra", "ngl")]
    + [f"s3.injection.seed{s}.{f}.full" for s in (303, 404) for f in ("wra", "ngl")]
    + [f"s3.injection.harnessG.{r}.{f}.{v}" for r in ("r0-9", "r10-19") for f in ("wra", "ngl")
       for v in ("preorigin", "full")])


DESIGN_V3 = [f"s3.injection.seed{s}.{f}.preorigin" for s in (505, 606) for f in ("wra", "ngl")]
DESIGN_V3B = [f"s3.injection.seed{s}.{f}.preorigin" for s in (515, 616) for f in ("wra", "ngl")]
DESIGN_V3C = [f"s3.injection.seed{s}.{f}.preorigin" for s in (525, 626) for f in ("wra", "ngl")]


def syn_members(units: list[str]) -> dict:
    return {u: {"none": SYN, "description": SYN} for u in units}


def table_members(unit: str, window: str, what: str) -> dict:
    return {unit: {"ids": ["gnss:all_wra_ngl"], "window": window, "description": what}}


LOOKS = {
    "diag_a1_trials": dict(
        units=SPENT_INJECTION, data_scope="gnss_injection_trials:2007-01..2026-09",
        purpose="diagnose", members=syn_members(SPENT_INJECTION),
        note="S3 repair round 3 (S3r), before any method change: anatomy of the archived "
             "injection trial tables of the failed tests (v1 seed 303, v2 seed 404, pre-origin "
             "and full; harness family G r0-9 and r10-19; tune r1-r3 seeds 101/202): which "
             "injected steps are missed or mis-sized, against size, sign, station noise, "
             "data density, gaps, nearby real steps, record ends and season; no new trial"),
    "diag_a1_background": dict(
        units=["s3.product_tables.v2_preorigin_base"],
        data_scope="gnss_preorigin_series:2007-01..2022-12", purpose="diagnose",
        members=table_members("s3.product_tables.v2_preorigin_base", "2007-01..2022-12",
                              "v2 pre-origin family runs (no injection): station noise, gaps, "
                              "steps and excursions used as covariates of the trial anatomy, "
                              "and the step-size estimator's error budget"),
        note="S3 repair round 3 (S3r), diagnosis: v2 pre-origin base runs with the registered "
             "v2 tuned parameters (tune_r3), station noise (sigma_hf, lag-1 rho, power at "
             "30-90 d), gap structure and base steps as covariates of the A1 failures, and a "
             "re-estimation of archived failed trials (seed 404) to split the size error into "
             "estimator terms; no new trial draw"),
    "diag_a2_pairs": dict(
        units=U.A2_UNITS, data_scope="gnss_wra+rings+piles:2020-01..2021-12",
        purpose="diagnose", prereg_id=V2, members=U.a2_members("2020-01..2021-12"),
        note="S3 repair round 3 (S3r), before any method change: why A2 failed in v1 and v2: "
             "per pair, the 2021 extra of the vendor-uncleaned, cleaned, ngl_offfan-, wra_hf- "
             "and on-fan-removed GNSS against the ring/pile, the steps and flags inside the "
             "January/December months of 2020-21, each common mode's own 2021 extra and the "
             "monthly series of the failing pairs (KTES, GFES, TKJS, YCES)"),
    "diag_a2_candidates_geometry": dict(
        units=["s3.a2.candidate_inventory"], data_scope="gnss+rings+piles+leveling:2019-01..2022-12",
        purpose="inventory",
        members={"s3.a2.candidate_inventory": {
            "ids": ["gnss:ngl_fan", "ring_or_pile:all"], "window": "2019-01..2022-12",
            "description": "geometry and month counts only (no value, no extra): which NGL "
                           "stations and which other piles lie within 1 km of a ring or pile "
                           "and have January/December months in 2020-21"}},
        note="S3 repair round 3 (S3r): inventory of candidate fresh A2 references by geometry "
             "and data availability only (distances, months with data); no height, compaction "
             "or extra value is read or computed"),
    "smoke_v3": dict(
        units=["s3.product_tables.v3_smoke"], data_scope="gnss_preorigin_series:2007-01..2022-12",
        purpose="smoke_test",
        members=table_members("s3.product_tables.v3_smoke", "2007-01..2022-12",
                              "v3 pre-origin family runs, crash and runtime test only"),
        note="S3 repair round 3 (S3r): crash and runtime test of the v3 pipeline (smoke_v3.py) "
             "on the pre-origin base runs with the stage-1 starting parameters; prints run "
             "times and step/excursion counts only; no injection, no A1/A2/A3 score"),
    "tune_v3": dict(
        units=DESIGN_V3, data_scope="gnss_injection:preorigin_background:2007-01..2022-12",
        purpose="design_time_seed", members=syn_members(DESIGN_V3),
        note="S3 repair round 3 (S3r): v3 design (dataprep_staging/s3_gnss/v3/tune.py), new "
             "design-time seeds 505 (stage 1: scan window x z_min x delta_min, 2/3 cm) and 606 "
             "(stage 2: local size window, 2/3/5/10 cm) on the pre-origin backgrounds, rules "
             "declared in the tune.py docstring before the run; output results/dataprep/"
             "reports/S3_evidence/repair_2026-10-18/tune_v3/"),
    "design_null": dict(
        units=["s3.design.null_positions.preorigin"],
        data_scope="gnss_preorigin_series:2007-01..2022-12", purpose="design_time_evidence",
        members=table_members("s3.design.null_positions.preorigin", "2007-01..2022-12",
                              "v3 pre-origin base runs: candidate local step-size models "
                              "evaluated at injection-eligible days without any injection"),
        note="S3 repair round 3 (S3r), design-time evidence (design_null.py): error of the OLS "
             "local step-size model (trend order 1-3, half-window 20-90 d) at every "
             "injection-eligible day of the step-free pre-origin residuals; no injection, no "
             "A1/A2 score; informs the design, never a test"),
    "design_null_cm": dict(
        units=["s3.design.null_positions.preorigin"],
        data_scope="gnss_preorigin_series:2007-01..2022-12", purpose="design_time_evidence",
        note="S3 repair round 3 (S3r), design-time evidence (design_null_cm.py): error of the "
             "OLS local step-size model (+-20/30 d) at every injection-eligible day of the "
             "step-free pre-origin residuals after candidate detection common modes (none, "
             "leave-one-out median, Huber-scaled median, 5-nearest-station median, scaled); no "
             "injection, no A1/A2 score; informs the design, never a test"),
    "design_null_level": dict(
        units=["s3.design.null_positions.preorigin"],
        data_scope="gnss_preorigin_series:2007-01..2022-12", purpose="design_time_evidence",
        note="S3 repair round 3 (S3r), design-time evidence (design_null_level.py, and an "
             "ad-hoc structure check of no-trend level differences on +-2..30 d): null error of "
             "the level-difference size model (+-7..20 d) and of the v2 model under candidate "
             "detection common modes (current, 3/5 nearest stations, NGL all-station median). "
             "DISCLOSED SLIP: both ran at about 01:33-01:36 CST, about one minute BEFORE this "
             "look was recorded; no injection, no A1/A2 score; design-time evidence only"),
    "smoke_v3b": dict(
        units=["s3.product_tables.v3_smoke"], data_scope="gnss_preorigin_series:2007-01..2022-12",
        purpose="smoke_test",
        note="S3 repair round 3 (S3r): crash and runtime test of the revised v3 pipeline (R1' "
             "level-difference size, R6 5-nearest-station WRA detection common mode) on the "
             "pre-origin base runs (smoke_v3.py); run times and step/excursion counts only; no "
             "injection, no A1/A2/A3 score"),
    "tune_v3b": dict(
        units=DESIGN_V3B, data_scope="gnss_injection:preorigin_background:2007-01..2022-12",
        purpose="design_time_seed", members=syn_members(DESIGN_V3B),
        note="S3 repair round 3 (S3r): v3 design run 2 (dataprep_staging/s3_gnss/v3/tune.py), "
             "new design-time seeds 515 (stage 1: scan window x z_min x delta_min, 2/3 cm) and "
             "616 (stage 2: step-size model, 2/3/5/10 cm) on the pre-origin backgrounds, rules "
             "declared in the tune.py docstring before the run; output results/dataprep/"
             "reports/S3_evidence/repair_2026-10-18/tune_v3b/"),
    "erln_trace": dict(
        units=["s3.injection.seed515.ngl.preorigin"],
        data_scope="gnss_injection:preorigin_background:2007-01..2022-12",
        purpose="design_time_seed", members=syn_members(["s3.injection.seed515.ngl.preorigin"]),
        note="S3 repair round 3 (S3r): after design run 2, trace of the ERLN design-unit trials "
             "(seed 515: 0 of 12 found): robust spread of the scan statistic by year, and three "
             "seed-515 ERLN injections re-run with the 12-step cap and with the cap lifted, "
             "then with the time-local null scale (R2b); design units only"),
    "tune_v3c": dict(
        units=DESIGN_V3C, data_scope="gnss_injection:preorigin_background:2007-01..2022-12",
        purpose="design_time_seed", members=syn_members(DESIGN_V3C),
        note="S3 repair round 3 (S3r): v3 design run 3 (tune.py; run 2's grids and rules with "
             "the time-local null scale R2b), new design-time seeds 525 (stage 1) and 626 "
             "(stage 2) on the pre-origin backgrounds; output results/dataprep/reports/"
             "S3_evidence/repair_2026-10-18/tune_v3c/"),
    "post_test_anatomy": dict(
        units=["s3.injection.seed707.wra.preorigin", "s3.injection.seed707.ngl.preorigin"]
        + U.A2_UNITS,
        data_scope="gnss_injection+rings+piles:2007-01..2022-12", purpose="diagnose",
        prereg_id="S3.gnss_clean_acceptance.v3",
        members={**syn_members(["s3.injection.seed707.wra.preorigin",
                                "s3.injection.seed707.ngl.preorigin"]),
                 **U.a2_members("2007-01..2022-12")},
        note="S3 repair round 3 (S3r), AFTER the registered v3 A1 score (WRA 0.897, NGL 0.873, "
             "both failed): anatomy of the seed-707 trial table (not found / size failures by "
             "size, station, noise, excursion) and of the v3 A2 second-look pairs against v2 "
             "(which steps moved TKJS, YCES, GFES in or out); diagnosis only, no new score"),
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

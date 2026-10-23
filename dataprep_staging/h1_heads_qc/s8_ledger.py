"""H1a ledger rows (plan 4.1, rule U6) for every number the WP report quotes.

Usage: s8_ledger.py PRODUCT_DIR
Each number id names its data sources with one role each (input, validator,
design_time_evidence). Sources are explicit files, directories or product versions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

HIST = "AMP_V2/data/wells"
NEW = "data_fetch_api/new_measurements/heads_202210_202610"
STN = "AMP_V2/data/fan_stations.parquet"
WRA = "data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet"
TOWNS = "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json"
OM = "data_fetch_api/new_measurements/pressure_hourly/openmeteo_era5_2026-10-12"
CODIS = "data_fetch_api/new_measurements/pressure_hourly/cwa_codis_2026-10-12"
AUDIT = "results/dataprep/_pilots/audit_2026-10-11/artefact_events.csv"
EV = "results/dataprep/reports/H1a_evidence"
EVAL = "results/twin/hybrid_datum_gate_innov/evaluation.csv"
RHO = "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json"
GAPS = "dataprep_staging/harness/gaps.py"


def rows_for(prod: str) -> list[dict]:
    pv = prod
    R = []

    def add(nid, src, role, note=""):
        R.append({"number_id": f"H1a.{nid}", "source": src, "role": role,
                  "product_version": pv, "note": note})

    # acceptance
    for k in ("inject.recall_paired", "inject.recall_step", "inject.spurious_rate",
              "inject.base_flag_rate"):
        add(k, HIST, "input", "pre-origin raw records the injections were added to")
        add(k, GAPS, "input", "family O generator (seed 1000*ord('O')+r)")
        add(k, f"{EV}/injection_O.json", "validator", "injection outcome record")
    add("audit_agreement.hist", HIST, "input", "raw records re-screened")
    add("audit_agreement.hist", AUDIT, "validator", "audit event list (2012-22)")
    add("audit_agreement.new", NEW, "input", "raw records re-screened (reported only)")
    add("audit_agreement.new", AUDIT, "validator", "audit event list (2023-26)")
    add("real_events.flagged", NEW, "input", "2024 records of the three wells")
    add("real_events.flagged", prod, "validator", "product flags in the event windows")
    for k in ("o1.pre.median_paired_diff", "o1.post.median_paired_diff"):
        add(k, HIST if ".pre." in k else NEW, "input", "hourly heads")
        add(k, OM, "input", "ERA5 surface pressure for the loading")
    for k in ("pressure_drop.pre.n_on_20hpa", "pressure_drop.post.n_on_20hpa",
              "pressure_drop.power"):
        add(k, OM, "input", "ERA5 surface pressure at the well cells")
        if "power" not in k:
            add(k, prod, "input", "offset episodes from events.csv")
    add("leakprobe.n_diff", prod, "input", "full build")
    add("leakprobe.n_diff", f"{EV}/leakprobe.json", "validator", "rebuild without 2023+ files")
    # reported checks
    for k in ("cwa.daily_rmse", "cwa.hourly_rmse", "cwa.fall_slope"):
        add(k, OM, "input", "ERA5 cell series")
        add(k, CODIS, "validator", "CWA station pressure (467480, 467490, 467270)")
    add("rescore.ratio_change", EVAL, "input", "frozen challenge predictions")
    add("rescore.ratio_change", prod, "input", "L1 monthly means (flags 0/1, C20)")
    # barometric design and results
    for k in ("baro.design_choice", "baro.epoch_split", "baro.se_inflation"):
        add(k, HIST, "design_time_evidence", "2012-22 split-half fits")
        add(k, f"{EV}/design_time/baro_methods.json" if k == "baro.design_choice"
            else f"{EV}/design_time/baro_epochs.json", "design_time_evidence",
            "design-time record")
    add("baro.n_accepted", HIST, "input", "2012-22 fits per logger epoch")
    add("baro.n_accepted", OM, "input", "ERA5 pressure")
    # metadata and counts
    for k in ("meta.swapped", "meta.zero_zero", "meta.label_repaired", "meta.layer_proposed",
              "meta.ground_height_zero"):
        add(k, STN, "input", "station table (non-personal columns)")
        if k in ("meta.swapped", "meta.label_repaired"):
            add(k, TOWNS, "validator", "MOI township polygons (taiwan-atlas)")
        if k == "meta.zero_zero":
            add(k, WRA, "validator", "WRA station list fetched 2026-09-30")
    for k in ("events.pre", "events.post", "coverage.monthly", "flags.daily",
              "merge.overlap_2022q4"):
        add(k, HIST, "input", "historical cache")
        add(k, NEW, "input", "new network")
    add("coverage.monthly", RHO, "input", "pre-origin rho(k) for sigma_obs")
    # resume round (2026-10-12 evening): after-the-fact diagnosis and provenance checks
    add("inject.miss_causes", HIST, "input", "pre-origin raw records, injections replayed")
    add("inject.miss_causes", f"{EV}/injection_O.json", "input", "missed injections")
    add("inject.miss_causes", f"{EV}/injection_misses_diagnosis.json", "validator",
        "after-the-fact attribution of each miss (reported only)")
    add("real_events.cause_09190132", prod, "validator",
        "flag 9 from a pre-origin step (2019-10-31, +18.9 m) carried to the record end")
    add("pressure_drop.flag1_episodes", prod, "validator",
        "the three episodes on >= 20 hPa falls are flag-1 co-located corrections")
    add("check.rehash_v1", prod, "validator",
        f"{EV}/check_L1_clean_v1.json (common.check, 0 failures)")
    add("looks.logged", "results/dataprep/preregistration/looks.jsonl", "validator",
        "H1a looks: history (3) and assert_can_score (7)")
    return R


def main():
    prod_dir = Path(sys.argv[1])
    prod = rel_to_repo(prod_dir)
    path = ledger.append_rows("H1a", rows_for(prod))
    print("ledger rows ->", path)


if __name__ == "__main__":
    main()

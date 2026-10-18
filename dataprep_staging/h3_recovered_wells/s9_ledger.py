"""Step 9: ledger rows of WP H3 (plan 4.1, U6): one row per source and reported number.

The ledger allows one role per source per number. The L1 v3 heads are the ``input`` of the
artificial-gap and snapshot numbers although their held-out values are also the truth (the
row note says so); the registered result records are the ``validator``. H2's registered run
tables re-read after the scoring are ``validator`` rows of an after-the-fact number, never a
gate.

Usage: python -I s9_ledger.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import ledger  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

L1 = "results/dataprep/heads/L1_clean/v3"
H2V4 = "results/dataprep/heads/L2_filled_preorigin/v4"
UNITS = "results/dataprep/validation_units/v3"
GRID = "results/twin_forward/datum_gate_innov.npz"
EV = "results/dataprep/reports/H3_evidence"
AMP = "AMP_V2/data/wells"
AMP_ST = "AMP_V2/data/fan_stations.parquet"
PARAMS = "results/dataprep/reports/H2_evidence/design/design_params.json"
PILOT = "results/dataprep/_pilots/head_gapfill_pilot/out/recovery_hist.csv"


def product_version() -> str:
    base = C.REPO / "results/dataprep" / C.PRODUCT
    hits = [f"{C.PRODUCT}/{d.name}" for d in sorted(base.glob("v*"),
                                                     key=lambda p: int(p.name[1:]))
            if (d / "validation.json").exists()
            and json.loads((d / "validation.json").read_text()).get("release") == "h3_v1"]
    if not hits:
        raise SystemExit("no packaged heads/recovered_wells version")
    return hits[-1]


def rows():
    pv = product_version()
    out = []

    def add(num, src, role, note, version=pv):
        out.append({"number_id": num, "source": src, "role": role, "product_version": version,
                    "note": note})

    for num in ("H3.inventory.counts", "H3.reconcile.counts", "H3.candidates"):
        add(num, L1, "input", "L1 v3 monthly heads 2012-2022 (window_end 2022-12) and "
                              "station_meta_preorigin (repaired coordinates, layer codes)")
        add(num, AMP, "input", "twin's L0 cache: the twin's own QC rule (read-only)")
        add(num, AMP_ST, "input", "twin's station table: raw layer codes and coordinates")
        add(num, UNITS, "input", "the 158 model wells")
        add(num, GRID, "input", "model grid (active cells)")
    add("H3.reconcile.counts", PILOT, "design_time_evidence",
        "the pilot's counts quoted by the plan, compared, not reused")
    probe = {"AMP_V2/data/wells": "cache through 2022-12-31",
             "chou-shui-data/data/ls_cache": "original delivery's raw vendor files",
             "data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet":
             "vendor record window (start/end_datetime_from_dataset, DisuseDate only)",
             "data_fetch_api/new_measurements/heads_202210_202610": "new-network station "
                                                                    "files (names only)",
             "chou-shui-data/data/raw/gw_timeseries_raw.csv": "provenance check: the file "
                                                              "is gap-filled",
             "chou-shui-data/data/gw_timeseries.csv": "curated 61-well file (membership)",
             "chou-shui-data/data/gw_stations.csv": "curated 61-well station table"}
    for src, note in probe.items():
        add("H3.rh_probe", src, "input", note)
    add("H3.rh_probe", L1, "input", "L1 v3 daily rows of 09030212 (once-daily, day_valid "
                                    "False) and record ends")
    add("H3.layer_review", L1, "input", "station_meta_preorigin (H1a proposals, depths) and "
                                        "C20 monthly heads 2012-2022")
    for kind in ("coverage", "rmse"):
        num = f"H3.d_wells.{kind}"
        add(num, L1, "input", "pre-origin C20 matrix and clean daily record; masked C20 "
                              "cells of the test wells are also the truth")
        add(num, PARAMS, "input", "frozen H2 bridge (form shape, bridged-share sds)")
        add(num, "dataprep_staging/h2_heads_fill/runner.py", "input", "H2 runner (unchanged)")
        add(num, "dataprep_staging/harness/gaps.py", "input", "seed rule 1000*ord('d')+r")
        add(num, f"{EV}/registered/{C.TEST_D}.json", "validator",
            "registered result (H3.recovery.v1)")
    add("H3.snapshot", L1, "input", "C20 values (truth) and L1 valid days 2012-2018; the "
                                    "once-daily rows of 09030212 2019-2022")
    add("H3.snapshot", AMP, "input", "raw 00:00 readings 2012-2018 (L0, read-only)")
    add("H3.snapshot", f"{EV}/registered/{C.TEST_SNAP}.json", "validator",
        "registered result (H3.recovery.v1)")
    add("H3.after_fact.h2_runs", "results/dataprep/reports/H2_evidence/run_tables",
        "validator", "H2 registered d (r10-19, 60-69) and S (r10-19) run tables restricted "
                     "to the H3 test wells; after the H3 scoring, never a gate")
    for src, note in ((L1, "C20 months 2012-2022 of 09030212 and its neighbours; its "
                           "once-daily rows 2019-2022"),
                      ("results/dataprep/reports/H3_evidence/after_fact/"
                       "snapshot_consistency.json", "after-the-fact record")):
        add("H3.after_fact.snapshot_consistency", src,
            "input" if src == L1 else "validator", note)
    add("H3.product.counts", f"{EV}/after_fact/snapshot_consistency.json", "input",
        "post-look hold-back of 09030212 (tightening, disclosed)")
    add("H3.product.counts", H2V4, "input", "calibration targets (flags 0-2) of the admitted "
                                            "wells")
    add("H3.product.counts", L1, "input", "once-daily months of 09030212; envelope months")
    add("H3.product.counts", f"{EV}/registered/{C.TEST_D}.json", "validator",
        "per-well admission of bridged months")
    add("H3.product.counts", f"{EV}/registered/{C.TEST_SNAP}.json", "validator",
        "admission of the once-daily months")
    add("H3.product.counts", f"results/dataprep/preregistration/{C.PREREG_ID}.json", "input",
        "admission rule (gate_rule)")
    return out


def main():
    r = rows()
    have = {(x["number_id"], x["source"]) for x in ledger.read_all() if x.get("wp") == C.WP}
    new = [x for x in r if (x["number_id"], x["source"]) not in have]
    if new:
        ledger.append_rows(C.WP, new)
    allr = ledger.read_all()
    print("added", len(new), "H3 rows", sum(1 for x in allr if x.get("wp") == C.WP),
          "conflicts", len(ledger.conflicts(allr)))


if __name__ == "__main__":
    main()

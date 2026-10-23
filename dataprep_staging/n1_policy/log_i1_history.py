"""Append a history-disclosure look for the I1 uses of 2012-22 data that were never logged.

Every I1 record (R1..R*, R4/Q0, G2) is a declaration with no logged look, yet their temporal
screens (--holdout-months 36) scored heads 2020-01..2022-12 at the calibration wells, and
every rung's leveling gate (and G2, seed 2027) scored site folds over the 2012-21 surveys.
This look records that history so later audits and registrations see it. Members are read
from the deliverable's well list and from the leveling panel's sid and datetime columns only
(no elevation). User ruling 2026-10-23.

Usage: python -I log_i1_history.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402
from harness import prereg3  # noqa: E402

PANEL = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
WELLS = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_wells.csv"
NOTE = ("History disclosure (user ruling 2026-10-23), logged by WP N1 on behalf of I1: the "
        "I1 ladder records are declarations without logged looks. (1) Temporal screens "
        "(--holdout-months 36, stage3_temporal.csv) of R0 (batch 8) and the rungs R1, R1b, "
        "R1c, R1d-dom, R1d-sw, R1e-pw, R1e-cn, R4, R4p, Q0 and R* scored heads 2020-01..2022-12 "
        "at the calibration wells, and the rung verdicts used them. (2) Every rung's leveling "
        "out-of-fold gate, the G2 fresh-partition leveling score (seed 2027) and each "
        "calibrate_coupled in-sample validate_against_leveling scored the 2012-21 surveys. "
        "Dated now because the exact times are not recoverable; the uses happened "
        "2026-10-11..2026-10-22.")


def members() -> dict:
    sids = sorted(pd.read_csv(WELLS, dtype={"sid": str}).sid.astype(str).unique())
    lev = pd.read_parquet(PANEL, columns=["sid", "datetime"])
    d = pd.to_datetime(lev.datetime)
    bm = sorted(lev[(d >= "2012-01-01") & (d < "2022-01-01")].sid.astype(str).unique())
    return {
        "I1:temporal_screens_2020_2022": {
            "description": "the deliverable's calibration wells in the I1 temporal-screen "
                           "window (heads 2020-01..2022-12)",
            "ids": sids, "window": "2020-01..2022-12"},
        "I1:leveling_gates_2012_2021": {
            "description": "leveling benchmarks surveyed 2012-01..2021-12 (the column fits' "
                           "and leveling gates' surveys)",
            "ids": bm, "window": "2012-01..2021-12"},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    m = members()
    for u, v in m.items():
        print(f"{u}: {len(v['ids'])} ids, window {v['window']}")
    if a.dry_run:
        return 0
    rec = prereg3.record_look(sorted(m), "heads+leveling:2012-01..2022-12", "I1",
                              "history_disclosure", members=m, note=NOTE)
    print("logged", rec.get("time") if isinstance(rec, dict) else rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

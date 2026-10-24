"""Test units of N1.policy_response.v1, from metadata only.

Reads the deliverable's well list (sid, layer, x, y) and the leveling panel's sid, datetime
and coordinate columns. No elevation and no head value is read. The unit file is written
once and hashed into the registration.

* ``N1P:lev_pre2012``: benchmarks on the active grid with >= 2 survey intervals of 0.5-1.5
  years whose both ends lie in 2007-01..2010-12 (electricity starts 2007-01; W3 looked at
  2011-21). Window 2007-01..2010-12.
* ``N1P:wells_m1618``: the 158 calibration wells in the masked window 2016-01..2018-12.
* ``N1P:lev_m1618``: benchmarks on the grid with a survey within 45 days of the model month
  2015-05, 2016-05, 2017-05, 2018-05 and 2019-05 (the four increments of the masked
  window; the 2016-18 surveys are withheld from the column). Window 2016-01..2018-12.

Usage: python -I units_policy.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR  # noqa: E402

PANEL = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
WELLS = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_wells.csv"
OUT = REPORTS_DIR / "N1_policy" / "units_policy.json"
PRE = ("2007-01-01", "2011-01-01")
MAY = [2015, 2016, 2017, 2018, 2019]


def survey_meta() -> pd.DataFrame:
    df = pd.read_parquet(PANEL, columns=["sid", "datetime", "x_3826", "y_3826"])
    df["datetime"] = pd.to_datetime(df.datetime)
    df["sid"] = df.sid.astype(str)
    return df.sort_values(["sid", "datetime"]).reset_index(drop=True)


def on_grid(meta: pd.DataFrame, cent: np.ndarray, dx: float) -> pd.Series:
    xy = meta.groupby("sid")[["x_3826", "y_3826"]].first()
    d = np.sqrt(((xy.to_numpy()[:, None, :] - cent[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return pd.Series(d <= dx, index=xy.index)


def pre2012_ids(meta: pd.DataFrame, ok: pd.Series) -> list[str]:
    w = meta[(meta.datetime >= PRE[0]) & (meta.datetime < PRE[1])]
    out = []
    for sid, g in w.groupby("sid"):
        if not ok.get(sid, False):
            continue
        yrs = np.diff(g.datetime.map(pd.Timestamp.toordinal).to_numpy()) / 365.25
        if ((yrs >= 0.5) & (yrs <= 1.5)).sum() >= 2:
            out.append(sid)
    return sorted(out)


def may_ids(meta: pd.DataFrame, ok: pd.Series) -> list[str]:
    out = []
    for sid, g in meta.groupby("sid"):
        if not ok.get(sid, False):
            continue
        d = pd.DatetimeIndex(g.datetime)
        if all((np.abs((d - pd.Timestamp(f"{y}-05-01")).days) <= 45).any() for y in MAY):
            out.append(sid)
    return sorted(out)


def build(cent: np.ndarray, dx: float) -> dict:
    meta = survey_meta()
    ok = on_grid(meta, cent, dx)
    wells = sorted(pd.read_csv(WELLS, dtype={"sid": str}).sid.astype(str).unique())
    return {
        "N1P:lev_pre2012": {
            "description": "leveling benchmarks on the active grid with >= 2 survey intervals "
                           "of 0.5-1.5 yr inside 2007-01..2010-12 (step B)",
            "ids": pre2012_ids(meta, ok), "window": "2007-01..2010-12"},
        "N1P:wells_m1618": {
            "description": "the deliverable's 158 calibration wells in the masked window "
                           "2016-01..2018-12 (step C, heads)",
            "ids": wells, "window": "2016-01..2018-12"},
        "N1P:lev_m1618": {
            "description": "leveling benchmarks on the active grid surveyed within 45 days of "
                           "May 2015, 2016, 2017, 2018 and 2019 (step C, subsidence; the "
                           "2016-18 surveys are withheld from the column fits)",
            "ids": may_ids(meta, ok), "window": "2016-01..2018-12"},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    import os

    os.chdir(REPO)
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths

    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)   # geometry only, no data
    units = build(grid.centroids(), grid.dx)
    for k, v in units.items():
        print(f"{k}: {len(v['ids'])} ids, window {v['window']}")
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if OUT.exists():
            raise SystemExit(f"{OUT} exists; written once")
        OUT.write_text(json.dumps({"units": units}, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

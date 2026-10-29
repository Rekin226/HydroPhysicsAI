"""N2C rehearsal readout: the recorded-pumping contrast c = PR - baseline of one forward run
at the 158 registered wells (engineering; model outputs and pre-origin well positions only).

Checks, for each ``--forward`` npz written with ``--recorded-pumping PR=...`` (and
optionally ``PRCH=...``): the contrast is exactly 0 in every month up to the forecast origin
(the hindcast and restart are untouched) and before the multiplier's first month; and it
reports the well-demeaned contrast RMS over the Changhua (station code 07) and Yunlin (09)
well-months of the rehearsal months, the number of months, and how many of the 74 Changhua
wells sit on a Changhua cell of the county map. The RMS sets EXPECTATIONS ONLY: X1 is a
post-origin product whose manifest forbids 'selection', 'model_design' and
'design_time_evidence', so nothing here chooses a model, an arm or a threshold.

No head observation is read: ``build_head_field`` is called with t1 = the origin month + 1
(the same pre-origin window ``twin.prospective register`` uses) only to place each well on
its cell and layer. Writes a NEW JSON (refuses to overwrite).

Usage: python -P rehearse_n2c.py --forward <npz> [--forward <npz> ...] --out <json>
       (PYTHONPATH = the frozen code checkout; cwd = the repository)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrophysics.twin.grid import FanGrid
from hydrophysics.twin.heads import build_head_field

STATIONS = "AMP_V2/data/fan_stations.parquet"
WELLS = "AMP_V2/data/wells"
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
ZR = "results/twin/prospective_2026-11/predictions.csv"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def well_cells(fw: dict, dates: pd.DatetimeIndex, origin: int):
    grid = FanGrid(int(fw["nx"]), int(fw["ny"]), float(fw["dx"]), float(fw["x0"]),
                   float(fw["y0"]), fw["mask"])
    hf = build_head_field(WELLS, pd.read_parquet(STATIONS), t0=str(dates[0].date()),
                          t1=str((dates[origin] + pd.offsets.MonthBegin()).date()))
    out = {}
    for i, xy in enumerate(hf.xy):
        c = grid.active_index(*xy)
        if c is not None:
            out[hf.sids[i]] = (c, int(hf.layers[i]) - 1)
    return out


def rms_demeaned(c: np.ndarray) -> float:
    """c (W, T): RMS of the per-well demeaned contrast over all well-months."""
    d = c - c.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(d ** 2)))


def readout(path: Path, ids: list[str], ch_flags: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as z:
        fw = {k: z[k] for k in z.files}
    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    origin = int(fw["origin"])
    names = [str(s) for s in fw["scenario_names"]]
    base = names.index("baseline")
    cells = well_cells(fw, dates, origin)
    missing = sorted(set(ids) - set(cells))
    rec = {"forward": str(path), "sha256": sha(path), "scenarios": names,
           "origin": str(dates[origin].date()), "wells_placed": len(set(ids) & set(cells)),
           "wells_missing": missing, "arms": {}}
    ch_ids = [s for s in ids if s.startswith("07") and s in cells]
    yl_ids = [s for s in ids if s.startswith("09") and s in cells]
    rec["changhua_wells_on_changhua_cells"] = int(sum(bool(ch_flags[cells[s][0]])
                                                      for s in ch_ids))
    rec["yunlin_wells_on_changhua_cells"] = int(sum(bool(ch_flags[cells[s][0]])
                                                    for s in yl_ids))
    hm = fw["heads_mean"]                                   # (S, L, A, T)
    for arm in names:
        if arm == "baseline" or not arm.startswith("PR"):
            continue
        k = names.index(arm)
        full = hm[k] - hm[base]                             # (L, A, T)
        pre = full[..., :origin + 1]
        rec_arm = {"max_abs_contrast_to_origin_all_cells": float(np.abs(pre).max()),
                   "exactly_zero_to_origin": bool((pre == 0).all())}
        post = dates > dates[origin]
        nz = np.flatnonzero(np.abs(full).reshape(-1, full.shape[-1]).max(axis=0) > 0)
        rec_arm["first_nonzero_month"] = (str(dates[nz[0]].date()) if len(nz) else None)
        win = post & (dates <= pd.Timestamp("2026-08-01"))
        for county, sel in (("changhua", ch_ids), ("yunlin", yl_ids)):
            c = np.stack([full[cells[s][1], cells[s][0], :][win] for s in sel])
            rec_arm[f"{county}_wells"] = len(sel)
            rec_arm[f"{county}_months"] = int(win.sum())
            rec_arm[f"{county}_rms_demeaned_m"] = rms_demeaned(c)
            rec_arm[f"{county}_mean_contrast_m"] = float(c.mean())
        rec["arms"][arm] = rec_arm
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forward", action="append", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; written once")
    ids = sorted(pd.read_csv(ZR, dtype={"station_id": str})["station_id"].unique())
    with np.load(COUNTY_MAP, allow_pickle=False) as z:
        ch = np.asarray(z["changhua"]).astype(bool)
    out = {"what": "N2C rehearsal (engineering; expectations only, never selection)",
           "zr_unit_ids": len(ids), "changhua_ids": sum(s.startswith("07") for s in ids),
           "yunlin_ids": sum(s.startswith("09") for s in ids),
           "county_map_sha256": sha(COUNTY_MAP),
           "runs": [readout(p, ids, ch) for p in a.forward]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

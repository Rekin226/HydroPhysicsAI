"""The twin's inputs, assembled exactly as ``hydrophysics.twin.inputs.load_twin_inputs`` does,
but reading the pump census with an explicit column list (plan 2.6: the census columns
戶名1 and 用電地址2 are never read; ``load_twin_inputs`` reads every column).

Every step calls the same repository functions in the same order (read-only imports), so
the arrays are identical; ``build_engine`` proves it against the physics scout's
``inputs_cwa.npz``, which ``load_twin_inputs`` produced.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The census columns ``clean_census``, ``energy_by_class`` and ``aggregate_pumps`` read.
CENSUS_COLS = ["sid", "PUMP_HP", "電號1", "PURPOSE", "TWD97_X", "TWD97_Y"]
FORBIDDEN_COLS = ("戶名1", "用電地址2")


def load_inputs(meta: dict, dx: float = 1000.0, verbose: bool = False):
    """``TwinInputs`` for a calibration's meta (``input_options``), census columns explicit.

    Mirrors ``load_twin_inputs(meter_filter=meta['meter_filter'], cap_duty=meta['cap_duty'],
    **input_options(meta))`` with the default paths, ``t0/t1`` 2012-01..2022-12 and
    ``backfill=True``. Must run with the repository root as the working directory (the
    default paths are relative)."""
    from hydrophysics.twin.calibrate_flow import (
        DEFAULT_PATHS,
        _ic_zone_map,
        _load_ground_elev,
        _load_recharge_field,
        prepare_series,
    )
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.heads import build_head_field
    from hydrophysics.twin.inputs import T0, T1, TwinInputs, input_options
    from hydrophysics.twin.pumping import clean_census
    from hydrophysics.twin.scenario import CLASSES, energy_by_class

    if Path.cwd().resolve() != REPO:
        raise RuntimeError(f"run from the repository root ({REPO}); the default paths are "
                           "relative")
    if any(c in CENSUS_COLS for c in FORBIDDEN_COLS):
        raise AssertionError("forbidden census column requested")
    opts = input_options(meta)
    meter_filter = meta.get("meter_filter", "dedupe-cap")
    cap_duty = float(meta.get("cap_duty", 1.0))
    t0, t1 = T0, T1
    P = dict(DEFAULT_PATHS)
    for k, v in P.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"{k}: {v!r} does not exist")
    grid = build_grid(P["polygon"], dx=dx)
    stn = pd.read_parquet(P["stations"])
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(P["wells_dir"], stn, t0=t0, t1=t1,
                          strict_coverage=bool(opts.get("strict_coverage", False)))
    idx, lay, raw, filled, xy, sids = [], [], [], [], [], []
    for w in range(len(hf)):
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        s = hf.heads[w]
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        raw.append(s)
        filled.append(prepare_series(s, backfill=True))
        xy.append(hf.xy[w])
        sids.append(str(hf.sids[w]))
    dates = pd.date_range(t0, t1, freq="MS", inclusive="left")
    ground_elev = _load_ground_elev(grid, stn, mode=opts.get("ground_elev", "wells"),
                                    dem_npz=opts.get("dem_npz", "results/twin/basemap.npz"),
                                    polygon=P["polygon"],
                                    log=(lambda m: print(m, flush=True)) if verbose else None)
    pumps = pd.read_parquet(P["pump_census"], columns=CENSUS_COLS)
    kwh = pd.read_parquet(P["pump_kwh"])
    if meter_filter != "none":
        pumps, kwh, _ = clean_census(
            pumps, kwh, cap_duty=(cap_duty if meter_filter == "dedupe-cap" else None),
            t0=t0, t1=t1)
    e_by_class, e_dates = energy_by_class(pumps, kwh, grid, t0, t1)
    assert len(e_dates) == len(dates)
    recharge = _load_recharge_field(grid, P["rf_timeseries"], P["rf_stations"],
                                    P["et_npz"], P["gw_stations"], t0, t1)
    zb = opts.get("zone_boundaries", "205,182")
    imp = bool(opts.get("ic_merged_proximal", False))
    return TwinInputs(grid=grid, hf=hf, dates=dates, obs_h=np.stack(raw),
                      obs_h_filled=np.stack(filled), obs_idx=np.array(idx, dtype="int64"),
                      obs_layer=np.array(lay, dtype="int64"),
                      well_xy=np.array(xy, dtype="float64"), sids=sids,
                      ground_elev=ground_elev,
                      E_by_class={k: e_by_class[k] for k in CLASSES if k in e_by_class},
                      recharge_field=recharge, stations=stn,
                      ic_zone_of_cell=_ic_zone_map(grid, zb) if imp else None,
                      ic_month0_filled=bool(opts.get("ic_month0_filled", False)),
                      ic_layered=bool(opts.get("ic_layered_proximal", False) and imp))


def input_files(meta: dict) -> tuple[list[str], list[str]]:
    """(files, directories) the loader reads, repository-relative."""
    from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
    from hydrophysics.twin.inputs import input_options

    opts = input_options(meta)
    files = [v for k, v in DEFAULT_PATHS.items() if k != "wells_dir"]
    if opts.get("ground_elev", "wells") != "wells":
        files.append(opts.get("dem_npz", "results/twin/basemap.npz"))
    return files, [DEFAULT_PATHS["wells_dir"]]

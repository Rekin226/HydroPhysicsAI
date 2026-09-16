"""One loader for everything the twin's forward and calibration paths consume.

``calibrate_flow.main`` assembled the grid, the QC'd head field, the ground elevation,
the pump electricity and the recharge field inline. The forward twin, the surrogate
builder and the 3D viewer need exactly the same objects, so they are assembled here once
and returned as a ``TwinInputs``. ``calibrate_flow`` keeps its own inline path so its
recorded provenance is untouched; the loader functions it calls are the ones reused here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .calibrate_flow import (
    DEFAULT_PATHS,
    _idw_initial_heads,
    _load_ground_elev,
    _load_recharge_field,
)
from .grid import FanGrid, build_grid
from .heads import HeadField, build_head_field
from .pumping import clean_census
from .scenario import CLASSES, energy_by_class

T0, T1 = "2012-01-01", "2023-01-01"


@dataclass
class TwinInputs:
    grid: FanGrid
    hf: HeadField
    dates: pd.DatetimeIndex            # month starts of the observed record (T months)
    obs_h: np.ndarray                  # (W, T) raw monthly heads, NaN where unobserved
    obs_h_filled: np.ndarray           # (W, T) interpolated for the calibration target
    obs_idx: np.ndarray                # (W,) active-cell index per well
    obs_layer: np.ndarray              # (W,) 0-indexed layer per well
    well_xy: np.ndarray                # (W, 2)
    sids: list[str]
    ground_elev: torch.Tensor          # (A,)
    E_by_class: dict[str, np.ndarray]  # class -> (A, T) kWh
    recharge_field: torch.Tensor       # (A, T) m/day, rain - ET0, clamped >= 0
    stations: pd.DataFrame = field(repr=False, default=None)

    @property
    def E_total(self) -> torch.Tensor:
        return torch.tensor(sum(self.E_by_class.values()), dtype=torch.float64)

    def initial_heads(self, month: int = 0, n_layers: int = 4,
                      well_mask: np.ndarray | None = None,
                      noise: np.ndarray | None = None) -> torch.Tensor:
        """Per-layer IDW field of the wells' heads observed in ``month`` -> (n_layers, A).

        ``well_mask`` restricts the wells (a fold's kept set); ``noise`` (W,) is added to
        the observed heads before interpolation, which is how the initial-condition
        ensemble gets spatially coherent perturbations rather than white noise per cell.
        """
        h = self.obs_h[:, month].copy()
        if noise is not None:
            h = h + noise
        sel = np.isfinite(h)
        if well_mask is not None:
            sel &= well_mask
        return _idw_initial_heads(self.grid, self.well_xy[sel], h[sel],
                                  self.obs_layer[sel], n_layers=n_layers)


def load_twin_inputs(paths: dict | None = None, dx: float = 1000.0,
                     t0: str = T0, t1: str = T1, wells_from: str | None = None,
                     meter_filter: str = "dedupe-cap", cap_duty: float = 1.0,
                     verbose: bool = True) -> TwinInputs:
    """Assemble the twin's inputs from the data cache. ``paths`` overrides DEFAULT_PATHS.
    ``meter_filter``/``cap_duty`` are ``calibrate_flow``'s census-cleaning options and must
    match the calibration the parameters came from."""
    P = dict(DEFAULT_PATHS)
    if paths:
        P.update({k: v for k, v in paths.items() if v is not None})
    for k, v in P.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"{k}: {v!r} does not exist -- see docs/DATA_FORMAT.md "
                                    "for how the cache is rebuilt")

    grid = build_grid(P["polygon"], dx=dx)
    stn = pd.read_parquet(P["stations"])
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(P["wells_dir"], stn, t0=t0, t1=t1)

    allowed = None
    if wells_from:
        allowed = set(pd.read_csv(wells_from)["sid"].astype(str))
    idx, lay, raw, filled, xy, sids = [], [], [], [], [], []
    for w in range(len(hf)):
        if allowed is not None and str(hf.sids[w]) not in allowed:
            continue
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        s = hf.heads[w]
        f = s if np.isfinite(s).all() else pd.Series(s).interpolate(
            limit_direction="both").to_numpy()
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        raw.append(s)
        filled.append(f)
        xy.append(hf.xy[w])
        sids.append(str(hf.sids[w]))
    if not idx:
        raise ValueError("no well fell inside the grid")

    dates = pd.date_range(t0, t1, freq="MS", inclusive="left")
    ground_elev = _load_ground_elev(grid, stn)
    pumps = pd.read_parquet(P["pump_census"])
    kwh = pd.read_parquet(P["pump_kwh"])
    if meter_filter != "none":
        pumps, kwh, _report = clean_census(
            pumps, kwh, cap_duty=(cap_duty if meter_filter == "dedupe-cap" else None),
            t0=t0, t1=t1)
    e_by_class, e_dates = energy_by_class(pumps, kwh, grid, t0, t1)
    assert len(e_dates) == len(dates)
    recharge = _load_recharge_field(grid, P["rf_timeseries"], P["rf_stations"],
                                    P["et_npz"], P["gw_stations"], t0, t1)
    if verbose:
        nan_frac = float(np.isnan(np.stack(raw)).mean())
        hp = {k: float(v.sum()) for k, v in e_by_class.items()}
        print(f"inputs: grid {grid.nx}x{grid.ny} @ {dx:.0f} m -> {grid.n_active} cells; "
              f"{len(hf)} wells passed QC, {len(idx)} inside the grid, "
              f"{100 * nan_frac:.2f}% NaN month-cells; classes "
              + ", ".join(f"{k} {v / 1e6:.1f} GWh" for k, v in hp.items()), flush=True)
    return TwinInputs(grid=grid, hf=hf, dates=dates, obs_h=np.stack(raw),
                      obs_h_filled=np.stack(filled), obs_idx=np.array(idx, dtype="int64"),
                      obs_layer=np.array(lay, dtype="int64"),
                      well_xy=np.array(xy, dtype="float64"), sids=sids,
                      ground_elev=ground_elev,
                      E_by_class={k: e_by_class[k] for k in CLASSES if k in e_by_class},
                      recharge_field=recharge, stations=stn)

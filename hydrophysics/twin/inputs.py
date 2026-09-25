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
    SW_COMPONENT_KEYS,
    _ic_zone_map,
    _idw_initial_heads,
    _load_ground_elev,
    _load_recharge_field,
    _merged_proximal_heads,
    prepare_series,
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
    # (A, T) m/day of canal irrigation deliveries (surface_water.py), or (K, A, T) with
    # --sw-components both; None unless a calibration used --sw-recharge and the caller
    # loaded it (load_sw_recharge / forward.attach_sw_recharge)
    sw_field: torch.Tensor | None = field(repr=False, default=None)
    # (A,) zone map when the calibration used --ic-merged-proximal: initial_heads then
    # replaces the proximal zone(s) with the merged-aquifer head, as calibrate_flow did
    ic_zone_of_cell: np.ndarray | None = field(repr=False, default=None)
    # --ic-layered-proximal (2026-09-25): keep the layers inside the proximal zone(s)
    ic_layered: bool = field(repr=False, default=False)
    # True only for an --ic-merged-proximal calibration WITHOUT --no-backfill: its month-0
    # field (and so its apex boundary head) was built from the back-filled heads of every
    # well; initial_heads(0) then does the same (review 2026-09-23: from the raw month-0
    # heads the proximal field lost the 07100/07080 wells and the apex head moved 25 m)
    ic_month0_filled: bool = field(repr=False, default=False)

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
        ``ic_month0_filled`` (see the field) takes month 0 from ``obs_h_filled``.
        """
        src = self.obs_h_filled if (month == 0 and self.ic_month0_filled) else self.obs_h
        h = src[:, month].copy()
        if noise is not None:
            h = h + noise
        sel = np.isfinite(h)
        if well_mask is not None:
            sel &= well_mask
        h0 = _idw_initial_heads(self.grid, self.well_xy[sel], h[sel],
                                self.obs_layer[sel], n_layers=n_layers)
        if self.ic_zone_of_cell is not None:
            h0, _ = _merged_proximal_heads(self.grid, h0, self.well_xy[sel], h[sel],
                                           self.ic_zone_of_cell,
                                           self.ic_zone_of_cell[self.obs_idx[sel]],
                                           layer_of=(self.obs_layer[sel] if self.ic_layered
                                                     else None))
        return h0


def input_options(meta: dict | None) -> dict:
    """``load_twin_inputs`` keyword arguments for the opt-in input constructions a
    calibration recorded in its meta (``calibrate_flow._input_opts_record``). Absent keys
    (every run before 2026-09-23) give the historical construction."""
    meta = meta or {}
    opts: dict = {}
    if meta.get("ic_merged_proximal"):
        opts["ic_merged_proximal"] = True
        opts["zone_boundaries"] = meta.get("zone_boundaries") or "205,182"
        # calibrate_flow built this run's h0 from back-filled month-0 heads unless
        # --no-backfill; the forward path must start (and pin the apex) from the same field
        if not meta.get("no_backfill"):
            opts["ic_month0_filled"] = True
        if meta.get("ic_layered_proximal"):
            opts["ic_layered_proximal"] = True
    ge = meta.get("ground_elev") or "wells"
    if ge != "wells":
        opts["ground_elev"] = ge
        opts["dem_npz"] = meta.get("ground_elev_dem_npz") or "results/twin/basemap.npz"
    if meta.get("strict_coverage"):
        opts["strict_coverage"] = True
    return opts


def load_twin_inputs(paths: dict | None = None, dx: float = 1000.0,
                     t0: str = T0, t1: str = T1, wells_from: str | None = None,
                     meter_filter: str = "dedupe-cap", cap_duty: float = 1.0,
                     verbose: bool = True, backfill: bool = True,
                     ic_merged_proximal: bool = False, zone_boundaries: str = "205,182",
                     ground_elev: str = "wells", dem_npz: str = "results/twin/basemap.npz",
                     strict_coverage: bool = False,
                     ic_month0_filled: bool = False,
                     ic_layered_proximal: bool = False) -> TwinInputs:
    """Assemble the twin's inputs from the data cache. ``paths`` overrides DEFAULT_PATHS.
    ``meter_filter``/``cap_duty`` are ``calibrate_flow``'s census-cleaning options and must
    match the calibration the parameters came from. ``backfill=False`` leaves
    ``obs_h_filled`` NaN wherever a month was never observed (``--no-backfill``).
    ``ic_merged_proximal``/``zone_boundaries``, ``ground_elev``/``dem_npz`` and
    ``strict_coverage`` are ``calibrate_flow``'s ``--ic-merged-proximal``,
    ``--ground-elev`` and ``--strict-coverage``; ``ic_month0_filled`` makes
    ``initial_heads(0)`` use the back-filled month-0 heads, as such a calibration did.
    ``ic_layered_proximal`` is ``--ic-layered-proximal`` (with ``ic_merged_proximal``).
    Pass ``**input_options(meta)``."""
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
    hf = build_head_field(P["wells_dir"], stn, t0=t0, t1=t1, strict_coverage=strict_coverage)

    allowed = None
    if wells_from:
        allowed = set(pd.read_csv(wells_from, dtype={"sid": str})["sid"])
    idx, lay, raw, filled, xy, sids = [], [], [], [], [], []
    for w in range(len(hf)):
        if allowed is not None and str(hf.sids[w]) not in allowed:
            continue
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        s = hf.heads[w]
        f = prepare_series(s, backfill=backfill)
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        raw.append(s)
        filled.append(f)
        xy.append(hf.xy[w])
        sids.append(str(hf.sids[w]))
    if not idx:
        raise ValueError("no well fell inside the grid")

    dates = pd.date_range(t0, t1, freq="MS", inclusive="left")
    ground_elev = _load_ground_elev(grid, stn, mode=ground_elev, dem_npz=dem_npz,
                                    polygon=P["polygon"],
                                    log=(lambda m: print(m, flush=True)) if verbose else None)
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
                      recharge_field=recharge, stations=stn,
                      ic_zone_of_cell=(_ic_zone_map(grid, zone_boundaries)
                                       if ic_merged_proximal else None),
                      ic_month0_filled=bool(ic_month0_filled),
                      ic_layered=bool(ic_layered_proximal and ic_merged_proximal))


SW_KEYS = ("sw_m_per_day", "dates", "dx", "n_active")


def load_sw_recharge(path: str, grid: FanGrid, dates: pd.DatetimeIndex,
                     components: str = "delivered") -> torch.Tensor:
    """Surface-water irrigation deliveries (``surface_water.py``'s npz) aligned to
    ``dates`` (month starts) -> ``(A, len(dates))`` float64 m/day of *delivered* water.

    ``components`` (``--sw-components``): ``"delivered"`` (``sw_m_per_day``, the
    historical field), ``"percolation"`` (``perc_m_per_day``, the paddy flood percolation
    potential of the v2 build) or ``"both"``, which returns ``(2, A, T)`` in that order.

    Checked, not trusted: the grid spacing and active-cell count must match the model's,
    every requested month must be present, and the field must be finite and >= 0. The
    recharge fraction is the calibration's ``log_sw_scale``, not part of this file.
    """
    if components not in SW_COMPONENT_KEYS:
        raise ValueError(f"components must be one of {tuple(SW_COMPONENT_KEYS)}, "
                         f"got {components!r}")
    keys = SW_COMPONENT_KEYS[components]
    if not os.path.exists(path):
        raise FileNotFoundError(f"--sw-recharge {path!r} not found -- build it with "
                                "python -m hydrophysics.twin.surface_water")
    with np.load(path, allow_pickle=False) as z:
        need = tuple(k for k in SW_KEYS if k != "sw_m_per_day") + keys
        missing = [k for k in need if k not in z]
        if missing:
            raise ValueError(f"{path}: missing key(s) {missing}; expected {need} (the "
                             "percolation field needs the surface_water.py v2 build)")
        fields = [np.asarray(z[k], dtype="float64") for k in keys]
        f_dates = pd.DatetimeIndex(pd.to_datetime([str(d) for d in z["dates"]]))
        dx, n_active = float(z["dx"]), int(z["n_active"])
    if abs(dx - float(grid.dx)) > 1e-6 or n_active != grid.n_active:
        raise ValueError(f"{path}: built for dx={dx:g} with {n_active} cells, the model "
                         f"grid is dx={grid.dx:g} with {grid.n_active}")
    for k, sw in zip(keys, fields, strict=True):
        if sw.shape != (grid.n_active, len(f_dates)):
            raise ValueError(f"{path}: {k} has shape {sw.shape}, expected "
                             f"({grid.n_active}, {len(f_dates)})")
        if not np.isfinite(sw).all() or (sw < 0).any():
            raise ValueError(f"{path}: {k} must be finite and >= 0")
    want = pd.DatetimeIndex(dates).to_period("M")
    have = f_dates.to_period("M")
    pos = have.get_indexer(want)
    if (pos < 0).any():
        raise ValueError(f"{path}: covers {have[0]}..{have[-1]}, missing "
                         f"{list(want[pos < 0].astype(str))[:6]}")
    if len(fields) == 1:
        return torch.tensor(fields[0][:, pos], dtype=torch.float64)
    return torch.tensor(np.stack([f[:, pos] for f in fields]), dtype=torch.float64)

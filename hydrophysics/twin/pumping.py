"""Registered-pump electricity census -> per-cell monthly abstraction.

Groundwater abstraction on this fan is unmetered, but electricity is not: the wisenvr
`etc-tpc-etc1mon-obs` dataset carries monthly kWh for 116,769 registered pumps, each
georeferenced and labelled by purpose. Energy converts to volume through the pump's own
hydraulics,

    Q = eta * E / (rho * g * lift),        lift = ground elevation - head,

so the only unknown is the wire-to-water efficiency ``eta`` -- one parameter per purpose
class, not a free spatiotemporal field. Because ``lift`` comes from the model's simulated
head, falling heads make the same electricity deliver less water, which is a real
energy-water feedback and not a modelling artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

RHO_G = 9800.0          # N/m3, rho * g for fresh water
J_PER_KWH = 3.6e6
MIN_LIFT_M = 2.0        # floor: a near-zero lift must not imply unbounded volume


def _cell_lookup(pumps: pd.DataFrame, grid) -> dict[str, int]:
    """Map ``sid`` -> active-cell index, dropping pumps outside the grid/mask.

    ``FanGrid.active_index`` recomputes ``mask[:row].sum()`` on every call, which is
    O(nx * ny) per lookup. Over ~116k pumps that adds up, so we precompute a cumulative
    active-cell count per row once here (without touching FanGrid's interface) and reuse
    it for every pump instead of calling ``active_index`` in a loop.
    """
    mask = grid.mask
    row_counts = mask.sum(axis=1)
    row_offset = np.concatenate([[0], np.cumsum(row_counts)[:-1]]).astype("int64")
    col_cumsum = np.cumsum(mask, axis=1)  # (ny, nx), running count of active cells to the left

    x = pd.to_numeric(pumps["TWD97_X"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(pumps["TWD97_Y"], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x) & np.isfinite(y)
    safe_x = np.where(finite, x, grid.x0)
    safe_y = np.where(finite, y, grid.y0)
    col = np.floor((safe_x - grid.x0) / grid.dx).astype("int64")
    row = np.floor((safe_y - grid.y0) / grid.dx).astype("int64")

    in_bounds = (
        finite
        & (row >= 0) & (row < grid.ny)
        & (col >= 0) & (col < grid.nx)
    )
    row_c = np.where(in_bounds, row, 0)
    col_c = np.where(in_bounds, col, 0)
    active = in_bounds & mask[row_c, col_c]

    idx = np.full(len(pumps), -1, dtype="int64")
    active_rows = row_c[active]
    active_cols = col_c[active]
    idx[active] = row_offset[active_rows] + col_cumsum[active_rows, active_cols] - 1

    sids = pumps["sid"].astype(str).to_numpy()
    return {sids[i]: int(idx[i]) for i in range(len(pumps)) if idx[i] >= 0}


def aggregate_pumps(pumps: pd.DataFrame, kwh: pd.DataFrame, grid,
                    t0: str, t1: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Monthly kWh summed into active grid cells -> ``((A, T) array, dates)``.

    Pumps whose coordinates fall outside the grid or the fan mask are dropped, never
    snapped to the nearest cell.
    """
    dates = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq="MS", inclusive="left")
    cell = _cell_lookup(pumps, grid)

    k = kwh.copy()
    k["datetime"] = pd.to_datetime(k["datetime"]).dt.to_period("M").dt.to_timestamp()
    k = k[(k["datetime"] >= dates[0]) & (k["datetime"] <= dates[-1])]
    k["cell"] = k["pump"].astype(str).map(cell)
    k = k.dropna(subset=["cell"])

    E = np.zeros((grid.n_active, len(dates)), dtype="float64")
    if k.empty:
        return E, dates
    tpos = {d: j for j, d in enumerate(dates)}
    k["tcol"] = k["datetime"].map(tpos)
    np.add.at(E, (k["cell"].astype(int).to_numpy(), k["tcol"].astype(int).to_numpy()),
              k["electricity_kwh"].to_numpy(dtype="float64"))
    return E, dates


def energy_to_volume(E: torch.Tensor, lift: torch.Tensor,
                     log_eta: torch.Tensor) -> torch.Tensor:
    """Convert period energy (kWh) and lift (m) to pumped volume (m3) for the period."""
    E = E.to(dtype=torch.float64)
    lift = lift.to(dtype=torch.float64)
    log_eta = log_eta.to(dtype=torch.float64)
    eta = torch.exp(log_eta)
    safe_lift = torch.clamp(lift, min=MIN_LIFT_M)
    return eta * E * J_PER_KWH / (RHO_G * safe_lift)

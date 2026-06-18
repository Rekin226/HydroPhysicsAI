"""Choushui head + subsidence explorer: grid/mask + interactive Plotly HTML.

Builds a self-contained HTML that animates the IDW observed-head surface and the
calibrated subsidence surface over the real fan, with a validation panel. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import numpy as np

from .data import GWData
from .subsidence import cumulative_drawdown, idw_interp, monthly_heads, well_xy


def head_grid(data: GWData, poly, n: int = 60):
    """IDW the monthly observed heads onto an n x n grid over the well bbox, masked to
    ``poly`` (a shapely polygon). Returns (XX, YY, HH, dates) with HH shape (Tm, n, n);
    cells outside the polygon are NaN.
    """
    from shapely.geometry import Point

    wxy = well_xy(data)
    xs = np.linspace(wxy[:, 0].min(), wxy[:, 0].max(), n)
    ys = np.linspace(wxy[:, 1].min(), wxy[:, 1].max(), n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)            # (n*n, 2)
    H, dates = monthly_heads(data)                              # (W, Tm)
    grid = idw_interp(pts, wxy, H)                              # (n*n, Tm)
    inside = np.array([poly.contains(Point(px, py)) for px, py in pts])
    grid[~inside] = np.nan
    HH = grid.T.reshape(len(dates), n, n)                      # (Tm, n, n)
    return XX, YY, HH, dates


def subsidence_grid(HH: np.ndarray, sk: float) -> np.ndarray:
    """Sk * cumulative drawdown per cell, from the (Tm, n, n) head history."""
    Tm = HH.shape[0]
    flat = HH.reshape(Tm, -1).T                                # (cells, Tm)
    D = cumulative_drawdown(flat)                              # (cells, Tm)
    return (sk * D).T.reshape(HH.shape)

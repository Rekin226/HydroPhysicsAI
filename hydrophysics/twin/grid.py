"""Fan polygon -> masked regular grid, with coordinate <-> cell lookups.

The Choushui fan covers ~2,144 km2. At 1 km that is ~2,100 active cells (8,400 four-layer
unknowns), which fits a full 132-month differentiable rollout on a 12 GB card. 500 m gives
~8,600 cells and is used only as a convergence check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from pyproj import Transformer


@dataclass
class FanGrid:
    """Regular grid over the fan's bounding box, masked to the polygon."""

    nx: int
    ny: int
    dx: float
    x0: float          # western edge of column 0, EPSG:3826 metres
    y0: float          # southern edge of row 0
    mask: np.ndarray   # (ny, nx) bool, True inside the fan

    @property
    def n_active(self) -> int:
        return int(self.mask.sum())

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, col) for a coordinate, or None if outside the grid or the mask."""
        col = int((x - self.x0) // self.dx)
        row = int((y - self.y0) // self.dx)
        if not (0 <= row < self.ny and 0 <= col < self.nx):
            return None
        if not self.mask[row, col]:
            return None
        return row, col

    def active_index(self, x: float, y: float) -> int | None:
        """Flat index into the active-cell vector (row-major over masked cells)."""
        rc = self.cell_of(x, y)
        if rc is None:
            return None
        row, col = rc
        return int(self.mask[:row].sum() + self.mask[row, :col].sum())

    def centroids(self) -> np.ndarray:
        """(n_active, 2) cell-centre coordinates in EPSG:3826 metres."""
        rows, cols = np.nonzero(self.mask)
        return np.column_stack([self.x0 + (cols + 0.5) * self.dx,
                                self.y0 + (rows + 0.5) * self.dx])


def build_grid(polygon_path: str, dx: float = 1000.0) -> FanGrid:
    """Read the fan GeoJSON (EPSG:4326), reproject to 3826, and mask a dx-metre grid."""
    with open(polygon_path) as fh:
        gj = json.load(fh)
    ring = np.array(gj["features"][0]["geometry"]["coordinates"][0], dtype="float64")[:, :2]
    tf = Transformer.from_crs(4326, 3826, always_xy=True)
    X, Y = tf.transform(ring[:, 0], ring[:, 1])
    poly = MplPath(np.column_stack([X, Y]))

    x0, y0 = float(np.min(X)), float(np.min(Y))
    nx = int(np.ceil((float(np.max(X)) - x0) / dx))
    ny = int(np.ceil((float(np.max(Y)) - y0) / dx))
    cx = x0 + (np.arange(nx) + 0.5) * dx
    cy = y0 + (np.arange(ny) + 0.5) * dx
    GX, GY = np.meshgrid(cx, cy)
    inside = poly.contains_points(np.column_stack([GX.ravel(), GY.ravel()]))
    return FanGrid(nx=nx, ny=ny, dx=dx, x0=x0, y0=y0,
                   mask=inside.reshape(ny, nx))

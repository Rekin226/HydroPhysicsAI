"""Open boundaries for the fan flow model: the coast and the mountain front.

Until 2026-09-11 the solver was a closed basin. ``flow._neighbour_index`` lists only the
faces between two *active* cells, so every face on the edge of the fan mask was no-flow:
nothing could leave to the sea and nothing could enter from the mountain front. The only
sink was pumping and the only source was recharge, and the monthly water balance had to
close through storage.

That is not the Choushui fan. It discharges to the Taiwan Strait along its western edge and
is fed at the apex where the Choushui river leaves the foothills. And it is what the
Stage-3 calibration had been telling us, run after run, in the language of pinned
parameters: with no boundary flux the optimiser can only balance water by shrinking both
forcings toward zero, so it floored the pump efficiency in every fold of the 2026-09-09
gate and, once the total-dynamic-head fix gave it a second lever, pinned that one at its
ceiling as well while cutting the recharge fraction from 0.93 to 0.36 (in-sample fit of
2026-09-11). Both knobs at opposite stops is an optimiser asking for net forcing ~ 0,
which a closed basin needs and an open one does not. It also matches the earlier
diagnosis that the model loses to IDW exactly in the shallow, forcing-dominated layers
and ties where IDW has to extrapolate.

This module supplies the geometry. Each boundary is a **general-head boundary**: a flux
``C * (h_b - h)`` into each boundary cell, with ``h_b`` a prescribed head and ``C`` a
conductance in m2/day that calibration learns. ``C -> 0`` recovers the closed basin, so
the open model contains the old one and the data decide.

- **coast**: the westernmost active cell of every grid row. ``h_b = 0`` (sea level, the
  same datum as the well elevations). One learnable ``C`` per layer: the shallow aquifer
  meets the sea directly while the deep confined ones extend offshore under aquitards.
- **apex**: the easternmost active cell of every row, inside the proximal zone.
  ``h_b`` is the initial IDW head at that cell, per layer (time-invariant: the mountain
  front is treated as a fixed far-field head). One learnable ``C`` shared by all layers,
  because the proximal zone is one merged aquifer in the zonal parameterisation.

The north and south edges of the fan (the Wu and Beigang rivers) stay no-flow; they are
surface-water divides, and adding them is a separate physical claim that should be tested
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import FanGrid

COAST_HEAD_M = 0.0     # sea level in the well-elevation datum


@dataclass(frozen=True)
class Boundaries:
    """Boundary cells and exposed-face counts, in active-cell index space."""

    coast_idx: np.ndarray      # (n_coast,) int64
    coast_faces: np.ndarray    # (n_coast,) float64, exposed west-facing faces per cell
    apex_idx: np.ndarray       # (n_apex,) int64
    apex_faces: np.ndarray     # (n_apex,) float64

    @property
    def n_coast(self) -> int:
        return int(self.coast_idx.size)

    @property
    def n_apex(self) -> int:
        return int(self.apex_idx.size)

    def describe(self) -> str:
        return (f"coast: {self.n_coast} cells / {self.coast_faces.sum():.0f} faces at "
                f"h_b = {COAST_HEAD_M:g} m; apex: {self.n_apex} cells / "
                f"{self.apex_faces.sum():.0f} faces at h_b = IDW initial head")


def _row_extremes(grid: FanGrid, side: str) -> np.ndarray:
    """Active-cell index of the westernmost (``side="west"``) or easternmost active cell
    in every grid row that has one.

    "Exposed west-facing face" alone is the wrong rule on this polygon: it also picks up
    interior notches and a detached sliver near the apex, which are not the coastline.
    The fan is a west-east band, so the coast is the first active cell scanning eastward
    along each row and the mountain front is the last.
    """
    rows, cols = np.nonzero(grid.mask)
    flat = np.arange(rows.size)
    out = []
    for r in np.unique(rows):
        sel = flat[rows == r]
        c = cols[sel]
        out.append(sel[np.argmin(c)] if side == "west" else sel[np.argmax(c)])
    return np.asarray(out, dtype="int64")


def fan_boundaries(grid: FanGrid, proximal_km: float = 205.0) -> Boundaries:
    """Locate the coast (westernmost active cell per row) and the apex (easternmost
    active cell per row, at easting >= ``proximal_km``) on the masked grid.

    The fan's west edge is the coastline and its east edge, inside the proximal zone, is
    the mountain front. Only the column direction is read, matching ``zones.fan_zones``,
    which bands the fan west-east on easting alone. Each boundary cell exposes exactly one
    face to its boundary, so the face count is one per cell.
    """
    x_km = grid.centroids()[:, 0] / 1000.0
    coast = _row_extremes(grid, "west")
    east = _row_extremes(grid, "east")
    apex = east[x_km[east] >= proximal_km]
    if coast.size == 0:
        raise ValueError("no coast cells: the grid has no active row")
    if apex.size == 0:
        raise ValueError(f"no apex cells: no easternmost active cell at easting >= "
                         f"{proximal_km} km")
    return Boundaries(coast_idx=coast, coast_faces=np.ones(coast.size, dtype="float64"),
                      apex_idx=apex, apex_faces=np.ones(apex.size, dtype="float64"))

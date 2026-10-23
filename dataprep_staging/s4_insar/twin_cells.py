"""WP S4a helper: lon/lat of the twin grid's cell centres, for coverage checks of the subsets.

Reads ``results/twin_forward/datum_gate_innov.npz`` (nx, ny, dx, x0, y0, mask; TWD97 / TM2,
EPSG:3826) and writes ``results/dataprep/insar_tools/s4a_run/twin_cells.npz`` with the cell
centres in EPSG:4326 (``lon``, ``lat``, shape (ny, nx)), the active ``mask`` and the grid hash.
The output stays under gitignored ``results/`` (study-area coordinates are never versioned).

Cell (i, j) centre: x = x0 + (j + 0.5) dx, y = y0 + (i + 0.5) dx, the convention of
``make_run_config.clip_box`` (whose box spans x0 .. x0 + nx dx and y0 .. y0 + ny dx).

Run with the hydro env: python -I twin_cells.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dataprep_staging (common)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
from common.hashing import grid_hash  # noqa: E402
from config import RUN_DIR, TWIN_GRID  # noqa: E402


def main() -> int:
    z = np.load(TWIN_GRID, allow_pickle=False)
    nx, ny, dx = int(z["nx"]), int(z["ny"]), float(z["dx"])
    x0, y0 = float(z["x0"]), float(z["y0"])
    mask = np.asarray(z["mask"]).astype(bool)
    if mask.shape != (ny, nx):
        raise SystemExit(f"mask shape {mask.shape} != (ny, nx) = {(ny, nx)}")
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    x = x0 + (jj + 0.5) * dx
    y = y0 + (ii + 0.5) * dx
    lon, lat = pyproj.Transformer.from_crs(3826, 4326, always_xy=True).transform(x, y)
    out = RUN_DIR / "twin_cells.npz"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, lon=lon, lat=lat, mask=mask,
             grid_hash=np.array(grid_hash(nx, ny, dx, x0, y0, z["mask"])))
    print(f"{mask.sum()} active of {mask.size} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

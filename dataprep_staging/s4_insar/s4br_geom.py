"""WP S4b repair round 2 (S4br): geometry of the second frame's clip window (069A_06632_162010)
on the LiCSBAS 2 x 2 multilooked grid, with round 1's rules (``s4b_geom.py``: rasters placed
by index on the window's reference geotransform; stable strip = land, DEM 100-600 m,
>= 2.5 km from active twin cells; the LiCSBAS15 mask is applied later), plus the index
offset of this grid against the primary frame's multilooked grid.

The window's reference geotransform comes from this round's fetch log
(``s4br_run/reference_grid_<frame>.json``: the frame's geo.E.tif geotransform and the clip
box's index window). Writes ``s4br_run/geom_<frame>.npz`` and a JSON summary. Licsbas env:

    dataprep_staging/s4_insar/licsbas_env.sh python -I dataprep_staging/s4_insar/s4br_geom.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
import s4b_config as C  # noqa: E402
import s4br_config as R  # noqa: E402
from s4b_geom import ml_mean, read_band  # noqa: E402
from s4b_lib import cell_index, twin_grid  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def main() -> int:
    frame = R.SECOND
    ref = json.loads((R.RUN_DIR / f"reference_grid_{frame}.json").read_text())
    gt, win = ref["geotransform"], ref["window"]
    lon0, dlon, lat0, dlat = gt[0] + win[0] * gt[1], gt[1], gt[3] + win[1] * gt[5], gt[5]
    ncol, nrow = int(win[2]), int(win[3])
    geoc = R.L0_SECOND / "GEOC"
    ras = {m: read_band(geoc / f"{frame}.geo.{m}.tif") for m in ("E", "N", "U", "hgt",
                                                                 "landmask")}
    for m, v in ras.items():
        if v.shape != (nrow, ncol):
            raise SystemExit(f"{m}: shape {v.shape} != window {(nrow, ncol)}")
    L, W = nrow // C.NLOOK, ncol // C.NLOOK
    jj, ii = np.meshgrid(np.arange(W), np.arange(L))
    lon = lon0 + (C.NLOOK * jj + C.NLOOK / 2) * dlon
    lat = lat0 + (C.NLOOK * ii + C.NLOOK / 2) * dlat
    tr = pyproj.Transformer.from_crs(4326, 3826, always_xy=True)
    x, y = tr.transform(lon, lat)
    mask, dx, x0, y0 = twin_grid()
    cell = cell_index(x, y, mask, dx, x0, y0)
    r, c = np.nonzero(mask)
    cen = np.column_stack([x0 + (c + 0.5) * dx, y0 + (r + 0.5) * dx])
    dist, _ = cKDTree(cen).query(np.column_stack([x.ravel(), y.ravel()]))
    dist_km = (dist / 1000.0).reshape(x.shape)
    E, N, U, hgt = (ml_mean(ras[m]) for m in ("E", "N", "U", "hgt"))
    land = ml_mean(np.where(np.isfinite(ras["landmask"]), ras["landmask"], 0.0)) >= 0.5
    stable_geom = (land & np.isfinite(hgt) & (hgt >= C.STABLE_MIN_HGT_M)
                   & (hgt <= C.STABLE_MAX_HGT_M) & (dist_km >= C.STABLE_MIN_DIST_KM)
                   & np.isfinite(U) & (U > 0))
    gp = np.load(C.RUN_DIR / f"geom_{C.PRIMARY}.npz", allow_pickle=False)
    plon0, pdlon, _, plat0, _, pdlat = gp["geotransform"]
    off_col = (lon0 - plon0) / (C.NLOOK * dlon)
    off_row = (lat0 - plat0) / (C.NLOOK * dlat)
    out = R.RUN_DIR / f"geom_{frame}.npz"
    np.savez_compressed(out, lon=lon, lat=lat, x=x, y=y, cell=cell, dist_active_km=dist_km,
                        E=E, N=N, U=U, hgt=hgt, land=land, stable_geom=stable_geom,
                        geotransform=np.array([lon0, dlon, 0.0, lat0, 0.0, dlat]),
                        window=np.array(win), nlook=np.array(C.NLOOK))
    summ = {"frame": frame, "ml_shape": [L, W], "native_shape": [nrow, ncol],
            "n_ml_pixels_in_active_cells": int((cell >= 0).sum()),
            "n_active_cells_with_ml_pixels": int(np.unique(cell[cell >= 0]).size),
            "n_active_cells": int(mask.sum()), "n_stable_geom_pixels": int(stable_geom.sum()),
            "ml_offset_vs_primary_cols_rows": [float(off_col), float(off_row)],
            "same_shape_as_primary_ml": bool(gp["x"].shape == x.shape), "out": str(out)}
    (R.RUN_DIR / f"geom_{frame}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

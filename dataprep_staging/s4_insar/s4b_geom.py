"""WP S4b: geometry of a frame's clip window on the LiCSBAS 2 x 2 multilooked grid.

Reads the clipped frame metadata rasters from L0 (E, N, U, hgt, landmask; all clipped with one
index window of the frame's reference grid by S4a) and the window's reference geotransform
from ``insar_licsar_subsets/v1/notes_for_S4b.json``. Every raster is placed by array index on
that geotransform (S4a caveat: some unw batches carry a stretched geotransform; LiCSBAS02
copies the first one it reads, so the geotransform in cum.h5 is ignored here).

Multilooked pixel (i, j) covers native rows 2i, 2i+1 and columns 2j, 2j+1 (LiCSBAS02 with
-n 2 drops a trailing odd row/column); its centre is lon0 + (2j + 1) dlon, lat0 + (2i + 1) dlat
with (lon0, lat0) the window's upper-left corner (pixel-is-area).

Writes ``results/dataprep/insar_tools/s4b_run/geom_<frame>.npz`` (ml grid: lon, lat, x, y
(EPSG:3826, m), cell (active twin-cell index or -1), dist_active_km, E, N, U, hgt, land,
stable_geom (land, hgt and distance rules of s4b_config; the LiCSBAS15 mask is applied later),
native grid: cell_native) and a JSON summary. Run with the licsbas env (GDAL):

    dataprep_staging/s4_insar/licsbas_env.sh python -I dataprep_staging/s4_insar/s4b_geom.py \
        --frame 069A_06694_131402
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
import s4b_config as C  # noqa: E402
from osgeo import gdal  # noqa: E402
from s4b_lib import cell_index, twin_grid  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

gdal.UseExceptions()


def frame_notes(frame: str) -> dict:
    return json.loads(C.S4A_NOTES.read_text())["frames"][frame]


def read_band(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(1)
    a = b.ReadAsArray().astype("float64")
    nd = b.GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    return a


def ml_mean(a: np.ndarray, n: int = C.NLOOK) -> np.ndarray:
    """n x n block mean ignoring NaN (LiCSBAS02 multilook layout; trailing rows/cols dropped)."""
    L, W = a.shape[0] // n, a.shape[1] // n
    b = a[:L * n, :W * n].reshape(L, n, W, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=[C.PRIMARY, C.DESC])
    a = ap.parse_args(argv)
    nf = frame_notes(a.frame)
    geoc = C.REPO / nf["geoc_dir"]
    lon0, dlon, _, lat0, _, dlat = nf["reference_geotransform_of_window"]
    win = nf["reference_grid"]["window"]           # col, row, ncol, nrow
    ncol, nrow = int(win[2]), int(win[3])
    ras = {m: read_band(geoc / f"{a.frame}.geo.{m}.tif") for m in ("E", "N", "U", "hgt",
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
    # native grid -> twin cell (for the native-resolution coherence statistic)
    nj, ni = np.meshgrid(np.arange(ncol), np.arange(nrow))
    xn, yn = tr.transform(lon0 + (nj + 0.5) * dlon, lat0 + (ni + 0.5) * dlat)
    cell_native = cell_index(xn, yn, mask, dx, x0, y0)
    out = C.RUN_DIR / f"geom_{a.frame}.npz"
    C.RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lon=lon, lat=lat, x=x, y=y, cell=cell, dist_active_km=dist_km,
                        E=E, N=N, U=U, hgt=hgt, land=land, stable_geom=stable_geom,
                        cell_native=cell_native.astype(np.int32),
                        geotransform=np.array([lon0, dlon, 0.0, lat0, 0.0, dlat]),
                        window=np.array(win), nlook=np.array(C.NLOOK))
    n_act = int(mask.sum())
    summ = {"frame": a.frame, "ml_shape": [L, W], "native_shape": [nrow, ncol],
            "n_ml_pixels_in_active_cells": int((cell >= 0).sum()),
            "n_active_cells_with_ml_pixels": int(np.unique(cell[cell >= 0]).size),
            "n_active_cells": n_act,
            "n_stable_geom_pixels": int(stable_geom.sum()),
            "stable_hgt_m_range": [float(np.nanmin(hgt[stable_geom])),
                                   float(np.nanmax(hgt[stable_geom]))]
            if stable_geom.any() else None,
            "U_ml_range": [float(np.nanmin(U)), float(np.nanmax(U))],
            "U_ml_median_active": float(np.nanmedian(U[cell >= 0])),
            "E_ml_median_active": float(np.nanmedian(E[cell >= 0])),
            "out": str(out)}
    (C.RUN_DIR / f"geom_{a.frame}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

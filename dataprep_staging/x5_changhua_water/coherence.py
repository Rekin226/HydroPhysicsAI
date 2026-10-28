"""Monthly LiCSAR short-pair coherence over each irrigation district's canal service cells.

Frame 069A_06694 (the primary ascending frame, 2020-01 onward), pairs with a temporal baseline
of at most 12 days, assigned to the calendar month of the pair's midpoint. A pixel counts for
a district when it lies in a twin cell (1 km, the zone_county_map centroids) whose X4
pre-origin canal service share is positive for that district and whose county matches
(Changhua cells for 'chc', the rest for 'yl'). Coherence is stored as uint8 x 255; 0 = no data.

Output: results/dataprep/x5_changhua_water/work/coherence_monthly.csv (month, district,
coh_mean, n_pairs, n_pixels). The L0 GeoTIFFs are read only; nothing is copied.

Usage: python -I coherence.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_changhua_water import x5const as K  # noqa: E402


def district_cells() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    zc = np.load(K.COUNTY_MAP)
    cen = np.asarray(zc["centroids"], dtype=float)
    ch = zc["changhua"].astype(bool)
    x4 = np.load(K.X4_PRE)
    return cen, {"chc": (np.asarray(x4["share_chc_service"]) > 0) & ch,
                 "yl": (np.asarray(x4["share_yl_service"]) > 0) & ~ch}


def main() -> int:
    cen, cells = district_cells()
    tree = cKDTree(cen)
    tr = Transformer.from_crs(4326, 3826, always_xy=True)
    files: dict[str, str] = {}
    for f in sorted(glob.glob(f"{K.LICSAR}/**/*.geo.cc.tif", recursive=True)):
        files.setdefault(os.path.basename(f).split(".")[0], f)
    cache: dict = {}
    rows = []
    for p, f in sorted(files.items()):
        a, b = p.split("_")
        da, db = pd.Timestamp(a), pd.Timestamp(b)
        if (db - da).days > K.SHORT_PAIR_DAYS:
            continue
        im = Image.open(f)
        tf = im.tag_v2
        nx, ny = im.size
        key = (nx, ny, tuple(tf.get(33922)), tuple(tf.get(33550)))
        if key not in cache:
            lon0, lat0 = tf.get(33922)[3], tf.get(33922)[4]
            d = tf.get(33550)[0]
            jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = tr.transform(lon0 + (jj + 0.5) * d, lat0 - (ii + 0.5) * d)
            _, idx = tree.query(np.c_[x.ravel(), y.ravel()])
            inside = ((np.abs(x.ravel() - cen[idx, 0]) <= 500)
                      & (np.abs(y.ravel() - cen[idx, 1]) <= 500))
            cell = np.where(inside, idx, -1).reshape(ny, nx)
            cache[key] = {k: (cell >= 0) & m[np.clip(cell, 0, None)] for k, m in cells.items()}
        arr = np.asarray(im).astype(float)
        arr[arr == 0] = np.nan
        arr /= 255.0
        mid = da + (db - da) / 2
        for k, m in cache[key].items():
            v = arr[m]
            n = int(np.isfinite(v).sum())
            rows.append({"pair": p, "mid": mid.strftime("%Y-%m-%d"),
                         "month": mid.strftime("%Y-%m"), "district": k,
                         "coh": float(np.nanmean(v)) if n else np.nan, "n_pixels": n})
    r = pd.DataFrame(rows)
    K.WORK.mkdir(parents=True, exist_ok=True)
    r.to_csv(K.WORK / "coherence_pairs.csv", index=False)
    g = (r.dropna(subset=["coh"]).groupby(["month", "district"])
         .agg(coh_mean=("coh", "mean"), n_pairs=("coh", "size"), n_pixels=("n_pixels", "mean"))
         .reset_index())
    g.to_csv(K.WORK / "coherence_monthly.csv", index=False)
    print(f"short pairs {r['pair'].nunique()}, district-months {len(g)}, "
          f"months {g['month'].min()}..{g['month'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

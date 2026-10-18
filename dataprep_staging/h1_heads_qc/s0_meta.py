"""H1a stage 0: station metadata repair and ERA5-cell assignment (run with python -I).

Usage: s0_meta.py WORKDIR
Writes WORKDIR/station_meta_clean.parquet and WORKDIR/meta_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import meta  # noqa: E402
from h1_heads_qc.fetch_openmeteo import CELLS  # noqa: E402


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    towns = meta.load_towns()
    clean, summ = meta.repair(towns=towns)
    # ERA5 cell: nearest grid point; stations without coordinates use their township centroid
    from pyproj import Transformer
    tr = Transformer.from_crs(3826, 4326, always_xy=True)
    cent = towns.assign(c=towns.geometry.centroid)
    cells, src = [], []
    for _, r in clean.iterrows():
        lat, lon, how = r.lat, r.lon, "station"
        if not np.isfinite(lat):
            m = cent[(r.TownName == cent.TOWNNAME) & (r.CountyName == cent.COUNTYNAME)]
            if len(m):
                lon, lat = tr.transform(m.c.iloc[0].x, m.c.iloc[0].y)
                how = "township_centroid"
            else:
                how = "none"
        if np.isfinite(lat):
            c = min(CELLS, key=lambda q: (q[0] - lat) ** 2 + (q[1] - lon) ** 2)
            cells.append(f"{c[0]:.2f}_{c[1]:.2f}")
        else:
            cells.append(None)
        src.append(how)
    clean["era5_cell"] = cells
    clean["era5_cell_from"] = src
    summ["era5_cells"] = clean.era5_cell.value_counts().to_dict()
    summ["era5_cell_from"] = clean.era5_cell_from.value_counts().to_dict()
    clean.to_parquet(work / "station_meta_clean.parquet", index=False)
    Path(work / "meta_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1,
              default=str))
    print(json.dumps({k: summ[k] for k in ("swapped", "era5_cells", "era5_cell_from",
                                           "layer_proposed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

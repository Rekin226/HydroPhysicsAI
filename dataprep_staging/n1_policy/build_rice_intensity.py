"""Per-cell paddy intensity for fix F2 (paddy-weighted irrigation return flow).

intensity(cell) = mean over 2012-2022 of the township's annual rice-planted area (crop 1 +
crop 2, ha; MOA, as transcribed in canal_redistribution_x4/full/v1
township_season_rice_fallow.csv) divided by the township's area (ha), assigned to each
active cell by the township containing its centroid (towns-10t_2021, X4's own reader).
Cells outside a listed township get 0. Pre-origin years only (2012-2022). Forcing data; no
head or leveling value is read. Written once.
Usage: results/twin_runs/par24/py.sh dataprep_staging/n1_policy/build_rice_intensity.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, REPO, rel_to_repo  # noqa: E402

SRC = DATAPREP / "canal_redistribution_x4/full/v1/township_season_rice_fallow.csv"
OUT = DATAPREP / "n1_policy/rice_intensity/v1"


def main() -> int:
    if (OUT / "rice_intensity.npz").exists():
        raise SystemExit(f"{OUT} exists; written once")
    os.chdir(REPO)
    from x4_canal_leveling.canal_build import TOWNS, pixel_town, towns_3826

    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths

    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    cent = grid.centroids()
    tw = towns_3826()
    tw["town"] = tw.COUNTYNAME + tw.TOWNNAME
    tid = pixel_town({"c": (cent[:, 0], cent[:, 1])}, tw)["c"]
    r = pd.read_csv(SRC)
    r = r[(r.year >= 2012) & (r.year <= 2022)]
    annual = r.groupby(["town", "year"]).rice_planted_ha.sum().groupby("town").mean()
    area_ha = (tw.geometry.area / 1e4).to_numpy()
    dens = np.array([annual.get(t, 0.0) for t in tw.town]) / area_ha
    inten = np.where(tid >= 0, dens[np.clip(tid, 0, None)], 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "rice_intensity.npz", share=inten, n_active=np.int64(len(cent)),
             town=np.array([tw.town.iloc[i] if i >= 0 else "" for i in tid]))
    sha = hashlib.sha256((OUT / "rice_intensity.npz").read_bytes()).hexdigest()
    meta = {"product": "n1_policy/rice_intensity", "version": "v1",
            "definition": __doc__.split("\n\n")[1].strip(),
            "inputs": [rel_to_repo(SRC), rel_to_repo(TOWNS)], "sha256": sha,
            "n_active": int(len(cent)), "cells_with_town": int((tid >= 0).sum()),
            "intensity_p50_p90_max": [float(np.quantile(inten, .5)),
                                      float(np.quantile(inten, .9)), float(inten.max())],
            "towns_with_rice": int((annual > 0).sum())}
    (OUT / "MANIFEST.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

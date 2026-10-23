"""N1's external test units, from station metadata only (no head value is read here).

* ``N1:ext_wells``: monitoring wells of the fan network that are NOT among the 158
  calibration wells of the deliverable (``stage3_datum_gate_cwa/stage3_wells.csv``), sit in an
  active cell of the twin grid and carry a layer code 1-4 (``heads/L1_clean/v3``
  ``station_meta_preorigin``; H1a's repaired coordinates). The deliverable's theta never saw
  them. ``site_disjoint`` marks wells whose site (``site_id``) holds no calibration well:
  the plan's site-grouped folds keep nested wells together, so only these are spatially held
  out in that sense.
* ``N1:ext_deep_yunlin``: the subset in Yunlin with layer code 3 or 4 (twin layers 2-3), the
  test wells of the public-well tests T2 and T3.
* ``N1:cal_wells``: the 158 calibration wells (consistency checks and the Phase 1-2 masked
  refits; never fresh for an in-sample fit).
* ``N1:mask_E2015`` (etc.): the calibration wells within an event window, the temporal unit a
  masked refit holds out.
* ``N1:piles``: the 7 printed deep-benchmark sensors (``deep_benchmarks_printed/v2``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import TWIN_GRID_NPZ  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402

TWIN_WELLS = K.TWIN_THETA.parent / "stage3_wells.csv"


def grid() -> dict:
    z = np.load(TWIN_GRID_NPZ, allow_pickle=False)
    g = {k: z[k] for k in ("nx", "ny", "dx", "x0", "y0", "mask")}
    g["mask"] = np.asarray(g["mask"], bool)
    # active index in row-major (y, x) order, as hydrophysics.twin.grid.Grid.active_index
    act = -np.ones(g["mask"].shape, dtype=np.int64)
    act[g["mask"]] = np.arange(int(g["mask"].sum()))
    g["active_idx"] = act
    return g


def cell_index(x: np.ndarray, y: np.ndarray, g: dict | None = None) -> np.ndarray:
    """Active-cell index of TWD97 points (-1 outside the active grid)."""
    g = g or grid()
    nx, ny, dx = int(g["nx"]), int(g["ny"]), float(g["dx"])
    j = np.floor((np.asarray(x, float) - float(g["x0"])) / dx).astype(int)
    i = np.floor((np.asarray(y, float) - float(g["y0"])) / dx).astype(int)
    ok = (j >= 0) & (j < nx) & (i >= 0) & (i < ny) & np.isfinite(x) & np.isfinite(y)
    out = -np.ones(len(j), dtype=np.int64)
    out[ok] = g["active_idx"][i[ok], j[ok]]
    return out


def station_meta() -> pd.DataFrame:
    cols = ["sid", "site_id", "CountyName", "TownName", "x_twd97", "y_twd97", "xy_flag",
            "layer_code", "layer_flag", "WellDepth", "EstablishDate"]
    return pd.read_parquet(K.HEADS_DIR / "station_meta_preorigin.parquet", columns=cols)


def calibration_wells() -> pd.DataFrame:
    return pd.read_csv(TWIN_WELLS, dtype={"sid": str})


def ext_wells() -> pd.DataFrame:
    m = station_meta()
    cal = calibration_wells()
    g = grid()
    m["cell"] = cell_index(m.x_twd97.to_numpy(), m.y_twd97.to_numpy(), g)
    m["is_cal"] = m.sid.isin(cal.sid)
    calsites = set(m.loc[m.is_cal, "site_id"])
    e = m[~m.is_cal & (m.cell >= 0) & m.layer_code.isin(["1", "2", "3", "4"])].copy()
    e["layer"] = e.layer_code.astype(int) - 1          # twin layer index (0-3)
    e["site_disjoint"] = ~e.site_id.isin(calsites)
    cxy = m.loc[m.is_cal, ["x_twd97", "y_twd97"]].to_numpy()
    d = np.sqrt(((e[["x_twd97", "y_twd97"]].to_numpy()[:, None, :] - cxy[None]) ** 2).sum(-1))
    e["dist_nearest_cal_km"] = d.min(axis=1) / 1000.0
    e["county"] = e.CountyName.map({"彰化縣": "Changhua", "雲林縣": "Yunlin",
                                    "南投縣": "Nantou"}).fillna(e.CountyName)
    return e.sort_values("sid").reset_index(drop=True)


def units() -> dict:
    e = ext_wells()
    cal = calibration_wells()
    g = grid()
    cal_cells = cell_index(cal.x.to_numpy(), cal.y.to_numpy(), g)
    deep_yl = e[(e.county == "Yunlin") & e.layer.isin([2, 3])]
    out = {
        "N1:ext_wells": {
            "description": "fan monitoring wells outside the deliverable's 158 calibration "
                           "wells, in active cells, layer code 1-4 (heads/L1_clean/v3 "
                           "station_meta_preorigin); heads 2013-01..2022-12",
            "ids": sorted(e.sid), "window": "2013-01..2022-12"},
        "N1:ext_deep_yunlin": {
            "description": "the Yunlin wells of N1:ext_wells with layer code 3-4; heads "
                           "2013-01..2022-12", "ids": sorted(deep_yl.sid),
            "window": "2013-01..2022-12"},
        "N1:cal_wells": {
            "description": "the deliverable's 158 calibration wells (stage3_wells.csv); heads "
                           "2013-01..2022-12", "ids": sorted(cal.sid),
            "window": "2013-01..2022-12"},
    }
    for ev, spec in K.EVENTS.items():
        y = spec["year"]
        out[f"N1:mask_{ev}"] = {
            "description": f"the 158 calibration wells in the masked window of {ev} "
                           f"({y}-01..{y}-05)", "ids": sorted(cal.sid),
            "window": f"{y}-01..{y}-05"}
    piles = pd.read_parquet(K.PILES_PRINTED / "printed_monthly.parquet",
                            columns=["sensor_id", "station", "depth_m"]).drop_duplicates()
    out["N1:piles"] = {
        "description": "the 7 printed deep-benchmark sensors of deep_benchmarks_printed/v2, "
                       "Jan-May 2020, 2021 and 2022", "ids": sorted(piles.sensor_id),
        "window": "2020-01..2022-05"}
    meta = {"n_ext": int(len(e)), "n_ext_site_disjoint": int(e.site_disjoint.sum()),
            "n_ext_layerflag0": int((e.layer_flag == 0).sum()),
            "n_ext_deep_yunlin": int(len(deep_yl)), "n_cal": int(len(cal)),
            "n_cal_cells_active": int((cal_cells >= 0).sum())}
    return {"units": out, "meta": meta}


if __name__ == "__main__":
    u = units()
    print(json.dumps(u["meta"], indent=1))
    e = ext_wells()
    print(pd.crosstab([e.county, e.site_disjoint], e.layer))
    print(e[(e.county == "Yunlin") & e.layer.isin([2, 3])][["sid", "site_id", "layer", "layer_flag",
                                                             "site_disjoint",
                                                             "dist_nearest_cal_km"]])

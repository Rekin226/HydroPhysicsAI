"""Gridding of the X2 county series and the P6 member reconstruction (read-only inputs).

* ``engine_grid``: active cells, cell counties and cell centres of the twin grid, from the
  P4 engine (``changhua_abstraction/engine/v2``, opened through the guard for
  ``model_design``; only grid keys are read).
* ``p6_member``: the release's P6 member (opened through the guard); its Taiwan Water cohort
  location weights (``cohort_weights__primary`` of ``tw_changhua_*`` / ``tw_yunlin_*``) are
  reused read-only, and the member is rebuilt cohort by cohort so that its Taiwan Water part
  and its other agencies' part can be separated (checked against ``rate_phys_m3d__primary``).
* ``cell_weights``: per county (and per Changhua system) weights over active cells for the
  variants county_uniform, p6_proxy and service_area. Weights sum to the in-grid fraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402

from x2_public_supply import x2const as K  # noqa: E402

GRID_KEYS = ("nx", "ny", "dx", "x0", "y0", "mask", "cell_county")


def engine_grid(use: str = "model_design") -> dict:
    p = guard.open_product(K.ENGINE, use)
    with np.load(p.path("engine_arrays.npz"), allow_pickle=False) as z:
        g = {k: z[k] for k in GRID_KEYS}
    mask = np.asarray(g["mask"], bool)
    ii, jj = np.nonzero(mask)                      # row-major over masked cells
    dx = float(g["dx"])
    g["xc"] = float(g["x0"]) + (jj + 0.5) * dx
    g["yc"] = float(g["y0"]) + (ii + 0.5) * dx
    g["n_active"] = int(mask.sum())
    g["cell_county"] = np.asarray(g["cell_county"]).astype(str)
    g["manifest"] = p.manifest
    return g


def p6_member(use: str) -> tuple[dict, dict]:
    p = guard.open_product(K.P6_MEMBER, use)
    z = np.load(p.path("sensitivity_member.npz"), allow_pickle=False)
    keys = ("dates", "dx", "n_active", "grid_hash", "mask_hash", "wra_to_model",
            "rate_phys_m3d__primary", "rate_phys_m3d__pending_tw_changhua", "cohort_id",
            "cohort_year", "cohort_volume_1e4m3_per_yr", "cohort_rate_phys_m3d",
            "cohort_weights__primary", "twin_layer_aquifer_code")
    return {k: z[k] for k in keys}, p.manifest


def rebuild_p6(m: dict) -> dict:
    """Cohort-by-cohort rebuild of the P6 primary (stop at July of the disposal year, layers
    3-4 split 0.5/0.5, as P6's member.py) and its Taiwan Water / other split."""
    dates = [str(d) for d in m["dates"]]
    T = len(dates)
    pos = {d: i for i, d in enumerate(dates)}
    ids = [str(c) for c in m["cohort_id"]]
    rate = np.asarray(m["cohort_rate_phys_m3d"], float)
    W = np.asarray(m["cohort_weights__primary"], float)
    stop = np.array([pos.get(f"{int(y):04d}-07", T) for y in m["cohort_year"]])
    active = (np.arange(T)[None, :] < stop[:, None]).astype(float)
    lw = np.array(K.LAYER_WEIGHTS)
    tw = np.array([c.startswith("tw_") for c in ids])
    parts = {}
    for name, sel in (("tw", tw), ("other", ~tw)):
        cell_t = (W[sel] * rate[sel][:, None]).T @ active[sel]
        parts[name] = lw[:, None, None] * cell_t[None, :, :]
    total = parts["tw"] + parts["other"]
    prim = np.asarray(m["rate_phys_m3d__primary"], float)
    err = float(np.abs(total - prim).max() / max(prim.max(), 1e-12))
    return {"dates": dates, "tw": parts["tw"], "other": parts["other"], "max_rel_err": err,
            "ids": ids, "tw_mask": tw, "rate": rate, "W": W, "stop": stop}


def county_cells(g: dict, county: str) -> np.ndarray:
    return g["cell_county"] == K.COUNTY_ZH[county]


def tw_weights_p6(m: dict, county: str) -> tuple[np.ndarray, list[str]]:
    ids = [str(c) for c in m["cohort_id"]]
    rows = [i for i, c in enumerate(ids) if c.startswith(f"tw_{county}_")]
    W = np.asarray(m["cohort_weights__primary"], float)[rows]
    if not np.allclose(W, W[0]):
        raise SystemExit(f"P6 tw_{county} cohorts do not share one weight vector")
    return W[0], [ids[i] for i in rows]


def town_of_cells(g: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Township of each active cell centre and each Changhua/Yunlin township's total area and
    active-grid area (km2), from the 2021 township polygons (EPSG:3826 = TWD97 TM2)."""
    import geopandas as gpd
    import shapely

    tw = gpd.read_file(K.TOWNS_TOPO, layer="towns")[["COUNTYNAME", "TOWNNAME", "geometry"]]
    tw = tw.set_crs(4326, allow_override=True).to_crs(3826)
    tw = tw[tw.COUNTYNAME.isin(list(K.COUNTY_ZH.values()))].reset_index(drop=True)
    tid = np.full(g["n_active"], -1, dtype=int)
    for i, geom in enumerate(tw.geometry):
        tid[shapely.contains_xy(geom, g["xc"], g["yc"])] = i
    info = pd.DataFrame({"county": tw.COUNTYNAME, "town": tw.TOWNNAME,
                         "area_km2": tw.geometry.area / 1e6,
                         "n_active_cells": np.bincount(tid[tid >= 0], minlength=len(tw))})
    info["active_area_km2"] = info.n_active_cells * (float(g["dx"]) / 1e3) ** 2
    return tid, info


def service_weights(g: dict, tid: np.ndarray, info: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per Changhua system: uniform over the active cells whose centre lies in its service
    townships; the weights sum to the in-grid share of the townships' area (capped at 1:
    the 1 km cells overfill a township's polygon area by up to about 1 %)."""
    out = {}
    ch = info[info.county == K.COUNTY_ZH["changhua"]]
    for sysname, towns in K.SERVICE_TOWNS.items():
        idx = [int(i) for i in ch.index[ch.town.isin(towns)]]
        missing = set(towns) - set(ch.town[ch.town.isin(towns)])
        if missing:
            raise SystemExit(f"{sysname}: towns not in the polygons {missing}")
        area = float(info.loc[idx, "area_km2"].sum())
        cells = np.isin(tid, idx)
        cell_area = (float(g["dx"]) / 1e3) ** 2
        frac = min(1.0, cells.sum() * cell_area / area)     # in-grid share of the area
        w = np.zeros(g["n_active"])
        w[cells] = frac / cells.sum()
        out[sysname] = w
    return out


def county_weights(g: dict, county: str, frac: float = 1.0) -> np.ndarray:
    """Uniform over the county's active cells (engine cell_county); the weights sum to
    ``frac`` (the in-grid share of the county's volume)."""
    c = county_cells(g, county)
    w = np.zeros(g["n_active"])
    w[c] = frac / c.sum()
    return w


def county_area_fraction(g: dict, info: pd.DataFrame, county: str) -> float:
    sel = info.county == K.COUNTY_ZH[county]
    return float(info.loc[sel, "active_area_km2"].sum() / info.loc[sel, "area_km2"].sum())


def grid_rates(months: list[str], county_rate: dict[str, np.ndarray],
               weights: dict[str, np.ndarray]) -> np.ndarray:
    """(4, A, T) m3/day: sum over keys of layer weights x cell weights x monthly rate."""
    lw = np.array(K.LAYER_WEIGHTS)
    A = len(next(iter(weights.values())))
    X = np.zeros((A, len(months)))
    for k, w in weights.items():
        X += np.outer(w, county_rate[k])
    return lw[:, None, None] * X[None, :, :]

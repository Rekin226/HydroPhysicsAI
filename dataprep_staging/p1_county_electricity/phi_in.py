"""WP P1 step 5 input: provisional in-grid fraction phi_in of Changhua county.

Plan P3 step 7 owns phi_in (MOI polygons and farmland, with its uncertainty, exported to P1
and P4). P3 has not run yet, so P1 computes the plan's "township check" itself, as a
provisional fallback that P3 supersedes:

* ``phi_area``    MOI township polygons (taiwan-atlas towns-10t 2021): Changhua area inside
                  the twin's active cells / total Changhua township area. (The first run used
                  the TW_COUNTY.gpkg county polygon, 0.790; the plan names MOI polygons, and
                  the planted-area weighting uses the same township polygons, so the area
                  version was switched to them. TW_COUNTY is kept as a reported check.)
* ``phi_planted`` township in-grid area fractions (MOI towns, taiwan-atlas towns-10t 2021)
                  weighted by township planted area (MOA 038 all crops, full-year rows, plus
                  J51 first and second rice crops; fallow, idle and mushrooms excluded),
                  survey years <= 2022 only (ROC 103, 104, 105, 110; U3).
* checks, reported only: the TW_COUNTY polygon fraction, and the label-based fraction
  (954 labelled cells x 1 km2 / TW_COUNTY area). Both town and county polygons include
  intertidal flats outside the active mask, which biases the area fractions low.

Central value: the geometric mean of phi_area and phi_planted. Uncertainty: the plan asks
for "at least the area-to-planted-area spread", so sigma_ln_phi = |ln phi_planted -
ln phi_area| (the whole spread taken as one sd).

The planted-area weighting assumes planted area is spread evenly over each township; a
farmland-based split inside the boundary townships is P3's job.

Usage: python -I phi_in.py <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "results/twin_forward/datum_gate_innov.npz"
COUNTY_GPKG = REPO / "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
TOWNS = REPO / "data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json"
MOA = REPO / "data_fetch_api/new_measurements/moa_crop_038"
J51 = REPO / "data_fetch_api/new_measurements/moa_rice_J51"
ROC_YEARS_PRE = ("103", "104", "105", "110")      # 2014, 2015, 2016, 2021 (<= 2022)
MUSHROOM = ("菇", "木耳", "靈芝", "蕈")
STEP = 50.0                                        # lattice spacing (m) inside each cell
CH = "彰化縣"

INPUTS = [GRID, COUNTY_GPKG, TOWNS] + [MOA / f"crop_{CH}_{y}.json" for y in ROC_YEARS_PRE] \
    + [J51 / f"rice_{y}.json" for y in ROC_YEARS_PRE]


def lattice(grid):
    mask = grid["mask"]
    dx, x0, y0 = float(grid["dx"]), float(grid["x0"]), float(grid["y0"])
    r, c = np.nonzero(mask)
    off = (np.arange(int(dx / STEP)) + 0.5) * STEP
    ox, oy = np.meshgrid(off, off)
    X = (x0 + c[:, None] * dx + ox.ravel()[None, :]).ravel()
    Y = (y0 + r[:, None] * dx + oy.ravel()[None, :]).ravel()
    cell = np.repeat(np.arange(r.size), off.size ** 2)
    return X, Y, cell


def planted_by_town():
    rows = []
    for y in ROC_YEARS_PRE:
        rows += json.load(open(MOA / f"crop_{CH}_{y}.json", encoding="utf-8"))
    d = pd.DataFrame(rows)
    d = d[(d["期作"] == "全年") & (d["縣市"] == CH)]
    d = d[~d["作物"].astype(str).str.contains("|".join(MUSHROOM))]
    d = d[~d["作物"].isin(["休閒面積", "荒廢面積"])]
    crop = d.groupby(["年度", "鄉鎮"])["種植面積(公頃)"].sum()
    r = []
    for y in ROC_YEARS_PRE:
        r += json.load(open(J51 / f"rice_{y}.json", encoding="utf-8"))
    r = pd.DataFrame(r)
    r = r[(r["縣市"] == CH) & r["期作"].isin(["一期作", "二期作"])]
    rice = r.groupby(["年度", "鄉鎮"])["種植面積"].sum()
    tot = crop.add(rice, fill_value=0.0)
    per_year = tot.groupby(level=0).sum()
    return tot.groupby(level=1).mean(), per_year, crop.groupby(level=1).mean(), \
        rice.groupby(level=1).mean()


def main(out_json: str) -> int:
    import geopandas as gpd
    import shapely

    grid = np.load(GRID, allow_pickle=True)
    X, Y, cell = lattice(grid)
    a_pt = STEP * STEP
    n_active = int(grid["mask"].sum())

    cty = gpd.read_file(COUNTY_GPKG).to_crs(3826)
    name_col = "NL_NAME_2" if "NL_NAME_2" in cty.columns else None
    if name_col is None:
        raise SystemExit(f"no NL_NAME_2 column in {COUNTY_GPKG}: {list(cty.columns)}")
    ch_geom = cty[cty[name_col].astype(str).str.replace("臺", "台") == CH].union_all()
    inside = shapely.contains_xy(ch_geom, X, Y)
    county_area = float(ch_geom.area)
    in_grid_area = float(inside.sum() * a_pt)
    phi_area = in_grid_area / county_area

    # label-based count, the twin's rule: centroid in polygon, else nearest polygon
    mask = grid["mask"]
    r, c = np.nonzero(mask)
    cx = float(grid["x0"]) + (c + 0.5) * float(grid["dx"])
    cy = float(grid["y0"]) + (r + 0.5) * float(grid["dx"])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cx, cy), crs=3826)
    j = gpd.sjoin(pts, cty[[name_col, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    lab = j[name_col].fillna("").astype(str).to_numpy()
    miss = np.flatnonzero(lab == "")
    if miss.size:
        near = gpd.sjoin_nearest(pts.iloc[miss], cty[[name_col, "geometry"]], how="left")
        near = near[~near.index.duplicated(keep="first")]
        lab[miss] = near[name_col].astype(str).to_numpy()
    n_label = int(np.sum(np.char.replace(lab.astype(str), "臺", "台") == CH))
    phi_label = n_label * float(grid["dx"]) ** 2 / county_area

    tw = gpd.read_file(TOWNS, layer="towns").set_crs(4326, allow_override=True).to_crs(3826)
    tw = tw[tw.COUNTYNAME.astype(str).str.replace("臺", "台") == CH].reset_index(drop=True)
    towns = []
    for _, row in tw.iterrows():
        g = row.geometry
        n_in = int(shapely.contains_xy(g, X, Y).sum())
        towns.append({"town": str(row.TOWNNAME), "area_km2": float(g.area) / 1e6,
                      "in_grid_km2": n_in * a_pt / 1e6})
    t = pd.DataFrame(towns).set_index("town")
    t["frac_in"] = (t["in_grid_km2"] / t["area_km2"]).clip(upper=1.0)
    planted, per_year, crop_m, rice_m = planted_by_town()
    t["planted_ha_mean"] = planted.reindex(t.index)
    unmatched = sorted(set(planted.index) - set(t.index))
    tt = t.dropna(subset=["planted_ha_mean"])
    phi_planted = float((tt["planted_ha_mean"] * tt["frac_in"]).sum()
                        / tt["planted_ha_mean"].sum())
    phi_town_area = float(t["in_grid_km2"].sum() / t["area_km2"].sum())
    phi_county_polygon = phi_area
    phi_area = phi_town_area
    phi_c = float(np.sqrt(phi_area * phi_planted))
    sig = float(abs(np.log(phi_planted) - np.log(phi_area)))
    out = {
        "status": "provisional (P1 fallback for plan P3 step 7; P3 supersedes)",
        "phi_in": phi_c,
        "sigma_ln_phi": sig,
        "phi_area": phi_area,
        "phi_planted": phi_planted,
        "phi_county_polygon_check": phi_county_polygon,
        "phi_label_cells": phi_label,
        "n_label_cells": n_label,
        "n_active_cells": n_active,
        "county_polygon_area_km2": county_area / 1e6,
        "county_polygon_in_grid_km2": in_grid_area / 1e6,
        "town_area_km2": float(t["area_km2"].sum()),
        "town_in_grid_km2": float(t["in_grid_km2"].sum()),
        "lattice_step_m": STEP,
        "planted_years_roc": list(ROC_YEARS_PRE),
        "planted_county_ha_by_year": {str(k): float(v) for k, v in per_year.items()},
        "towns_unmatched_in_survey": unmatched,
        "towns_without_survey": sorted(set(t.index) - set(planted.index)),
        "rule": "phi_in = sqrt(phi_area * phi_planted); sigma_ln_phi = |ln phi_planted - "
                "ln phi_area| (the whole area-to-planted spread as one sd, plan P3 step 7)",
        "inputs": [str(p.relative_to(REPO)) for p in INPUTS],
        "towns": t.reset_index().round(6).to_dict(orient="records"),
    }
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "towns"}, ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

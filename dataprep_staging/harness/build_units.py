"""Build ``results/dataprep/validation_units/v<N>/``: the design-time and fresh validation
units, shaped stress regions and fresh folds of plan WP A1, from repository data, and declare
the fresh units in a hashed pre-registration before any downstream test uses them.

    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh python -m harness.build_units   (from dataprep_staging/)

Inputs (read-only): the twin grid (``results/twin_forward/datum_gate_innov.npz``), the fan
polygon, the county polygons (``TW_COUNTY.gpkg``), the station table (columns ``sid``,
``GroundwaterZoneIdentifier``, ``GroundwaterLayerCode``, ``LocationByTWD97``, ``CountyName``
only), the well records 2012-01..2022-12 (the calibration well set is rebuilt with
``hydrophysics.twin.heads.build_head_field`` exactly as ``load_twin_inputs`` does), the
leveling panel (``sid``, dates, ``elev_m`` and coordinates; benchmarks selected as
``mechanism.py`` does, surveys 2012-2022), and the physics pilot's ``inputs_cwa.npz`` as a
cross-check of the well set and the cell counties. Nothing dated after 2022-12 is read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.hashing import grid_hash, mask_hash  # noqa: E402
from common.paths import PREREG_DIR, REPO, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402

from harness import __version__, coverage, prereg, units  # noqa: E402

sys.path.insert(0, str(REPO))
from hydrophysics.twin import leveling as lev  # noqa: E402
from hydrophysics.twin.grid import FanGrid, build_grid  # noqa: E402
from hydrophysics.twin.heads import build_head_field  # noqa: E402

POLYGON = REPO / "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json"
COUNTIES = REPO / "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
WELLS_DIR = REPO / "AMP_V2/data/wells"
LEVELING = REPO / "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
PILOT_INPUTS = REPO / "results/dataprep/_pilots/physics_inversion_hooks/inputs_cwa.npz"
STN_COLS = ["sid", "GroundwaterZoneIdentifier", "GroundwaterLayerCode", "LocationByTWD97",
            "CountyName"]
CH, YL = "彰化縣", "雲林縣"
T0, T1 = "2012-01-01", "2023-01-01"
#: outputs of the pilots that scored the design-time units; the design-time look seed is
#: dated to the earliest of their modification times (preserved by A0's move), so that the
#: seed is never later than the first documented look
PILOT_LOOK_EVIDENCE = [
    "physics_inversion_hooks/e_changhua_inversion.json",
    "physics_inversion_hooks/d_pseudo_changhua.json",
    "physics_inversion_hooks/f_temporal_holdout.json",
    "design_physics_inversion/p2_pseudo_region.json",
    "design_physics_inversion/p2_pseudo.json",
    "design_bayesian_fusion/p1b_pseudo.json",
    "design_ml_transfer/a2_scores.csv",
]
#: region names and sizes printed by the Bayesian-fusion pilot (p1b.log, first line)
FUSION_PILOT_SIZES = {"east_t1": 374, "east_t2": 385, "east_t3": 399, "north_t1": 379,
                      "north_t2": 360, "north_t3": 419, "east_h1": 574, "east_h2": 584,
                      "north_h1": 558, "north_h2": 600, "km1_0": 337, "km1_2": 384}


def pilot_look_time() -> tuple[str, str]:
    """(UTC time in the look-log format, file) of the earliest pilot look evidence."""
    import datetime as dt

    from common.paths import PILOTS
    best = min((PILOTS / p).stat().st_mtime for p in PILOT_LOOK_EVIDENCE)
    src = next(p for p in PILOT_LOOK_EVIDENCE if (PILOTS / p).stat().st_mtime == best)
    t = dt.datetime.fromtimestamp(best, dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), src


def load_grid() -> tuple[FanGrid, dict]:
    z = np.load(TWIN_GRID_NPZ, allow_pickle=False)
    g = FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]), float(z["y0"]),
                np.asarray(z["mask"], bool))
    gp = build_grid(str(POLYGON), dx=1000.0)
    same = (g.nx, g.ny, g.dx, g.x0, g.y0) == (gp.nx, gp.ny, gp.dx, gp.x0, gp.y0) and \
        np.array_equal(g.mask, gp.mask)
    return g, {"grid_npz_equals_polygon_grid": bool(same),
               "grid_hash": grid_hash(g.nx, g.ny, g.dx, g.x0, g.y0, g.mask),
               "mask_hash": mask_hash(g.mask), "n_active": g.n_active}


def cell_counties(cen: np.ndarray):
    import geopandas as gpd
    from shapely.geometry import Point
    g = gpd.read_file(COUNTIES).to_crs(3826)
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in cen], crs=3826)
    j = gpd.sjoin(pts, g[["NL_NAME_2", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    cc = j["NL_NAME_2"].fillna("").to_numpy().astype(object)
    miss = np.flatnonzero(cc == "")
    if miss.size:
        near = gpd.sjoin_nearest(pts.iloc[miss], g[["NL_NAME_2", "geometry"]], how="left")
        near = near[~near.index.duplicated(keep="first")]
        cc[miss] = near["NL_NAME_2"].to_numpy()
    try:
        gc = g[g.NL_NAME_2 == CH].geometry.union_all()
        gy = g[g.NL_NAME_2 == YL].geometry.union_all()
    except AttributeError:  # older geopandas
        gc = g[g.NL_NAME_2 == CH].geometry.unary_union
        gy = g[g.NL_NAME_2 == YL].geometry.unary_union
    border = gc.boundary.intersection(gy.boundary)
    return np.array([str(c) for c in cc]), border, int(miss.size)


def calibration_wells(grid: FanGrid) -> pd.DataFrame:
    stn = pd.read_parquet(STATIONS, columns=STN_COLS)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(str(WELLS_DIR), stn, t0=T0, t1=T1)
    county = stn.set_index("sid")["CountyName"]
    rows = []
    for w in range(len(hf)):
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        sid = str(hf.sids[w])
        rows.append({"sid": sid, "x": float(hf.xy[w, 0]), "y": float(hf.xy[w, 1]),
                     "cell": int(i), "layer": max(int(hf.layers[w]) - 1, 0),
                     "county_name": str(county.get(sid, ""))})
    return pd.DataFrame(rows)


def leveling_benchmarks(grid: FanGrid) -> pd.DataFrame:
    panel = pd.read_parquet(LEVELING, columns=["sid", "datetime", "elev_m", "x_3826",
                                                "y_3826"])
    panel = panel.rename(columns={"x_3826": "x", "y_3826": "y"})
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel = panel.sort_values(["sid", "datetime"]).reset_index(drop=True)
    sub = lev.site_subsidence(panel, T0, T1, min_obs=5, max_rate=0.5)
    xy = lev.site_xy(panel)
    rows = []
    for sid in sorted(sub):
        x, y = xy[sid]
        i = grid.active_index(x, y)
        if i is None:
            continue
        rows.append({"sid": str(sid), "x": x, "y": y, "cell": int(i),
                     "last_survey": str(sub[sid].index.max().date())})
    return pd.DataFrame(rows)


def cell_unit(uid, role, mask, wells, definition, group, extra=None):
    cells = np.flatnonzero(mask)
    win = mask[wells.cell.to_numpy()]
    w = wells[win]
    u = {"id": uid, "role": role, "kind": "cells", "group": group, "definition": definition,
         "n_cells": int(cells.size), "cells": [int(c) for c in cells],
         "n_wells": int(win.sum()), "wells": w.sid.tolist(),
         "n_wells_by_county": {k: int(v) for k, v in w.county.value_counts().items()}}
    if extra:
        u.update(extra)
    return u


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    t_start = time.time()
    grid, gchk = load_grid()
    cen = grid.centroids()
    x_km, y_km = cen[:, 0] / 1000.0, cen[:, 1] / 1000.0
    cc, border, n_nearest = cell_counties(cen)
    ch, yl = cc == CH, cc == YL
    wells = calibration_wells(grid)
    wells["county"] = wells.county_name.map({CH: "CH", YL: "YL"}).fillna("other")
    wells["cell_county"] = [{"彰化縣": "CH", "雲林縣": "YL"}.get(cc[c], "other")
                            for c in wells.cell]
    bm = leveling_benchmarks(grid)
    bm["county"] = [{"彰化縣": "CH", "雲林縣": "YL"}.get(cc[c], "other") for c in bm.cell]
    print(f"grid {grid.n_active} cells; wells {len(wells)}; benchmarks {len(bm)}; "
          f"{time.time() - t_start:.0f}s", flush=True)

    # ---- cross-check against the physics pilot's arrays (pre-origin pilot)
    pz = np.load(PILOT_INPUTS, allow_pickle=False)
    xchk = {
        "cell_county_equal": bool(np.array_equal(cc, pz["cell_county"].astype(str))),
        "well_sids_equal_in_order": bool(list(pz["sids"].astype(str)) == wells.sid.tolist()),
        "well_xy_max_abs_diff_m": float(np.max(np.abs(pz["well_xy"] - wells[["x", "y"]]
                                                       .to_numpy()))),
        "obs_idx_equal": bool(np.array_equal(pz["obs_idx"], wells.cell.to_numpy())),
        "obs_layer_equal": bool(np.array_equal(pz["obs_layer"], wells.layer.to_numpy())),
        "well_county_equal": bool(list(pz["well_county"].astype(str)) ==
                                  wells.county_name.tolist()),
        "pilot_grid_mask_equal": bool(np.array_equal(pz["mask"], grid.mask)),
        "cells_assigned_by_nearest_polygon": n_nearest,
    }
    dist_border = units.distance_to_line_km(cen, border)

    U = []
    # ---- design-time units
    ter = units.easting_terciles(x_km, yl)
    for k in ("west", "mid", "east"):
        U.append(cell_unit(f"dt.tercile.{k}", "design_time", ter[k], wells,
                           "Yunlin easting tercile (cuts at the 1/3, 2/3 quantiles of Yunlin "
                           f"cell eastings, {ter['_cuts_km']} km)", "dt.terciles"))
    hv = units.halves(y_km, yl)
    for k in ("north", "south"):
        U.append(cell_unit(f"dt.half.{k}", "design_time", hv[k], wells,
                           f"Yunlin {k} half (northing median {hv['_cut_km']:.3f} km; north "
                           "includes the median row); the north half is the river-adjacent "
                           "shaped stress region", "dt.halves"))
    fus = units.fusion_pilot_regions(x_km, y_km, yl)
    for k, m in fus.items():
        U.append(cell_unit(f"dt.fusion.{k}", "design_time", m, wells,
                           "Bayesian-fusion pilot region (p1b_pseudo.py: terciles, halves, "
                           "k-means(3) seeds 1-3, near-duplicates dropped)", "dt.fusion"))
    zon = units.changhua_zones(dist_border, ch, x_km)
    for k, m in zon.items():
        U.append(cell_unit(f"dt.zone.{k}", "design_time", m, wells,
                           "Changhua zone of P4 (Z1 x >= 205 km; Z2 within 10 km of the "
                           "Changhua-Yunlin county border, outside Z1; Z3 the rest)",
                           "dt.zones"))
    U.append({"id": "dt.block.2020_2022", "role": "design_time", "kind": "months",
              "group": "dt.blocks", "definition": "the pilots' temporal hold-out",
              **units.temporal_block("2020-01", "2022-12")})
    for county, sel in (("CH", wells.county == "CH"), ("YL", wells.county == "YL")):
        f = units.pilot_site_folds(wells[["x", "y"]].to_numpy(), sel.to_numpy(), 5, 0)
        U.append({"id": f"dt.folds.pilot_site_{county}", "role": "design_time",
                  "kind": "folds", "group": "dt.folds",
                  "definition": "the pilots' site-fold rule (coordinates rounded to 100 m, "
                                "default_rng(0) shuffle, round-robin into 5 folds) applied to "
                                "the county's calibration wells; the pilots applied the same "
                                "rule to the wells of each pseudo-region and of Changhua",
                  "folds": [{"fold": i, "test": wells.sid.iloc[ix].tolist()}
                            for i, ix in enumerate(f)]})
    # ---- fresh units
    rot = units.rotated_terciles(x_km, yl)
    for i in (1, 2, 3):
        U.append(cell_unit(f"fresh.rot_tercile.{i}", "fresh", rot[f"rot{i}"], wells,
                           "easting tercile with cut points rotated by half a tercile width: "
                           f"cuts at the 1/6, 1/2, 5/6 quantiles ({rot['_cuts_km']} km); unit 3 "
                           "is the wrap-around remainder (far west and far east strips)",
                           "fresh.rot_terciles", {"contiguous": i != 3}))
    for k, parts, prim in (("ME", ("mid", "east"), True), ("WM", ("west", "mid"), False)):
        m = ter[parts[0]] | ter[parts[1]]
        U.append(cell_unit(f"fresh.two_tercile.{k}", "fresh", m, wells,
                           f"two adjacent design-time terciles ({' + '.join(parts)}); "
                           + ("PRIMARY two-tercile region (closest Yunlin share to "
                              "Changhua's 74 of 158 wells)" if prim else
                              "secondary two-tercile region (declared, not primary)"),
                           "fresh.two_tercile", {"primary": prim}))
    km = units.kmeans_partition(cen / 1000.0, yl, 5, units.KMEANS_SEED)
    for k, m in km.items():
        U.append(cell_unit(f"fresh.{k.replace('km2027_', 'km2027.')}", "fresh", m, wells,
                           "5-region disjoint k-means partition of Yunlin cells (centroids in "
                           "km, KMeans(5, n_init=10, random_state=2027)); numbered west to "
                           "east by cluster-centre easting", "fresh.km2027"))
    U.append({"id": "fresh.block.2016_2018", "role": "fresh", "kind": "months",
              "group": "fresh.blocks", "definition": "mid-record temporal block",
              **units.temporal_block("2016-01", "2018-12")})
    band = units.river_band(dist_border, yl, x_km, 10.0, units.PROXIMAL_X_KM)
    U.append(cell_unit("fresh.river_band_YL10", "fresh", band, wells,
                       "Yunlin cells within 10 km of the Changhua-Yunlin county border and "
                       "west of x = 205 km: the mirror of Changhua's Z2 rule (shaped stress "
                       "region)", "stress"))
    folds_out = {}
    for name, sel, buf in (("wells_CH", wells.county == "CH", 5.0),
                           ("wells_YL", wells.county == "YL", 5.0),
                           ("wells_all", wells.county != "", 5.0)):
        w = wells[sel.to_numpy()]
        f = units.spatial_block_folds(w[["x", "y"]].to_numpy(), w.sid.tolist(), n_folds=5,
                                      diameter_km=10.0, buffer_km=buf, seed=units.FOLD_SEED)
        folds_out[name] = f
        U.append({"id": f"fresh.folds.{name}", "role": "fresh", "kind": "folds",
                  "group": "fresh.folds",
                  "definition": "site-grouped spatial-block folds: sites within 100 m, "
                                "complete-linkage clusters of diameter <= 10 km, 5 folds "
                                f"(seed {units.FOLD_SEED}), buffer {buf} km", "folds":
                  f["folds"], "params": f["params"]})
    lev_alt = {}
    for name, sel in (("leveling_CH", bm.county == "CH"), ("leveling_all", bm.county != "")):
        b = bm[sel.to_numpy()]
        f = units.spatial_block_folds(b[["x", "y"]].to_numpy(), b.sid.tolist(), n_folds=None,
                                      diameter_km=10.0, buffer_km=10.0, seed=units.FOLD_SEED)
        folds_out[name] = f
        # the rejected alternative (five balanced folds, same buffer), kept as evidence
        f5 = units.spatial_block_folds(b[["x", "y"]].to_numpy(), b.sid.tolist(), n_folds=5,
                                       diameter_km=10.0, buffer_km=10.0, seed=units.FOLD_SEED)
        lev_alt[name] = {"test_sizes": [len(fd["test"]) for fd in f5["folds"]],
                         "train_sizes": [len(fd["train"]) for fd in f5["folds"]]}
        U.append({"id": f"fresh.folds.{name}", "role": "fresh", "kind": "folds",
                  "group": "fresh.folds",
                  "definition": "leveling benchmarks (>= 5 surveys 2012-2022, max rate 0.5 "
                                "m/yr, inside the grid) in site-grouped spatial-block folds: "
                                "10 km complete-linkage clusters, leave one cluster out, "
                                "buffer 10 km (five balanced folds with the same buffer "
                                f"leave {min(lev_alt[name]['train_sizes'])} training "
                                "benchmarks in the worst fold; validation.json "
                                "leveling_5fold_alternative)",
                  "folds": f["folds"], "params": f["params"]})

    by = {u["id"]: u for u in U}

    # ---- overlaps of fresh cell units with every design-time cell unit
    masks = {u["id"]: np.isin(np.arange(grid.n_active), u["cells"]) for u in U
             if u["kind"] == "cells"}
    dt_ids = [k for k in masks if by[k]["role"] == "design_time"]
    overlaps = {}
    for k in masks:
        if by[k]["role"] != "fresh":
            continue
        js = {d: units.jaccard(masks[k], masks[d]) for d in dt_ids}
        top = max(js, key=js.get)
        overlaps[k] = {"max_jaccard": js[top], "with": top}
        by[k]["max_jaccard_design_time"] = {"value": js[top], "with": top}

    # ---- validation checks (expected values from the plan where it states them)
    def cnt(uid, key="n_cells"):
        return by[uid][key]

    checks = []

    def check(name, expected, measured, ok):
        checks.append({"check": name, "expected": expected, "measured": measured,
                       "pass": bool(ok)})

    check("grid npz equals the polygon grid (build_grid)", True,
          gchk["grid_npz_equals_polygon_grid"], gchk["grid_npz_equals_polygon_grid"])
    for k in ("cell_county_equal", "well_sids_equal_in_order", "obs_idx_equal",
              "obs_layer_equal", "well_county_equal", "pilot_grid_mask_equal"):
        check(f"physics pilot cross-check: {k}", True, xchk[k], xchk[k])
    check("physics pilot cross-check: well xy max |diff| (m)", 0.0,
          xchk["well_xy_max_abs_diff_m"], xchk["well_xy_max_abs_diff_m"] == 0.0)
    check("cells by county (CH, YL)", [954, 1158], [int(ch.sum()), int(yl.sum())],
          [int(ch.sum()), int(yl.sum())] == [954, 1158])
    nw = wells.county.value_counts().to_dict()
    check("calibration wells (total, CH, YL)", [158, 74, 84],
          [len(wells), nw.get("CH", 0), nw.get("YL", 0)],
          [len(wells), nw.get("CH", 0), nw.get("YL", 0)] == [158, 74, 84])
    check("well county equals its cell's county", 158,
          int((wells.county == wells.cell_county).sum()),
          bool((wells.county == wells.cell_county).all()))
    t3 = [cnt(f"dt.tercile.{k}") for k in ("west", "mid", "east")]
    check("design-time terciles: cells (plan 374/385/399)", [374, 385, 399], t3,
          t3 == [374, 385, 399])
    nh = [cnt("dt.half.north"), cnt("dt.half.north", "n_wells")]
    check("north half: cells, wells (plan 600, 37)", [600, 37], nh, nh == [600, 37])
    tt = [cnt("fresh.two_tercile.WM", "n_wells"), cnt("fresh.two_tercile.ME", "n_wells")]
    check("two-tercile regions: wells (plan 55 or 63 of 84)", [55, 63], tt, tt == [55, 63])
    zc = [[cnt(f"dt.zone.{z}"), cnt(f"dt.zone.{z}", "n_wells")] for z in ("Z1", "Z2", "Z3")]
    check("Changhua zones Z1/Z2/Z3: cells, wells (plan 139/13, 300/24, 515/37)",
          [[139, 13], [300, 24], [515, 37]], zc, zc == [[139, 13], [300, 24], [515, 37]])
    fsz = {k: int(m.sum()) for k, m in fus.items()}
    check("fusion pilot regions: names and cell counts equal the pilot's (p1b.log)",
          FUSION_PILOT_SIZES, fsz, fsz == FUSION_PILOT_SIZES)
    for grp, ids in (("rotated terciles", [f"fresh.rot_tercile.{i}" for i in (1, 2, 3)]),
                     ("k-means 2027", [u for u in by if u.startswith("fresh.km2027.")])):
        stack = np.stack([masks[i] for i in ids])
        disjoint = bool((stack.sum(0) <= 1).all())
        covers = bool(np.array_equal(stack.any(0), yl))
        check(f"{grp}: disjoint partition of the Yunlin cells", True,
              {"disjoint": disjoint, "covers_yunlin": covers}, disjoint and covers)
    mj = max(v["max_jaccard"] for v in overlaps.values())
    check("no fresh cell unit is a near-duplicate of a design-time region (Jaccard < 0.9)",
          "< 0.9", mj, mj < 0.9)
    dt_blk = set(by["dt.block.2020_2022"]["months"])
    fr_blk = set(by["fresh.block.2016_2018"]["months"])
    check("fresh temporal block: 36 months, disjoint from the 2020-22 hold-out",
          [36, 0], [len(fr_blk), len(fr_blk & dt_blk)], len(fr_blk) == 36 and not fr_blk & dt_blk)
    for name, f in folds_out.items():
        ids = list(f["fold_of"])
        cover = sorted(sum((fd["test"] for fd in f["folds"]), [])) == sorted(ids)
        site_ok = all(len({f["fold_of"][i] for i in ids if f["site_of"][i] == s}) == 1
                      for s in set(f["site_of"].values()))
        src = wells if name.startswith("wells") else bm
        pos = src.set_index("sid")[["x", "y"]]
        mind = []
        for fd in f["folds"]:
            if not fd["test"] or not fd["train"]:
                continue
            a = pos.loc[fd["test"]].to_numpy()
            b = pos.loc[fd["train"]].to_numpy()
            mind.append(float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min() / 1000))
        buf = f["params"]["buffer_km"]
        ok = cover and site_ok and all(m > buf for m in mind)
        check(f"fresh folds {name}: partition, sites kept whole, train-test distance > "
              f"{buf} km", True, {"n": len(ids), "covers": cover, "sites_whole": site_ok,
                                  "fold_sizes": [len(fd["test"]) for fd in f["folds"]],
                                  "train_sizes": [len(fd["train"]) for fd in f["folds"]],
                                  "min_train_test_km": [round(m, 3) for m in mind]}, ok)
    passed = all(c["pass"] for c in checks)
    seed_time, seed_src = pilot_look_time()
    if dry:
        print("design-time look seed", seed_time, seed_src)
        print("leveling 5-fold alternative", lev_alt)
        for c in checks:
            print(("PASS " if c["pass"] else "FAIL ") + c["check"], "->", c["measured"])
        print("overlaps", json.dumps(overlaps, ensure_ascii=False))
        print("units", [(u["id"], u.get("n_cells"), u.get("n_wells")) for u in U])
        print(f"dry run: nothing written; gate would be passed={passed}")
        return 0 if passed else 1

    # ---- write the product
    import sklearn
    d = manifest.new_version_dir("validation_units")
    registry = {
        "schema": "dataprep-validation-units/1", "created_by": "WP A1",
        "harness_version": __version__, "sklearn_version": sklearn.__version__,
        "grid": gchk, "origin": "2022-12",
        "membership_rule": "a well belongs to a cell unit when its model cell does",
        "roles": {"fresh": "validation units declared before any E1-E11 / P4.3 run; never "
                           "used for design choices",
                  "design_time": "used by the 2026-10-12 pilots for design choices; results "
                                 "on them are design-time evidence only"},
        "units": U}
    (d / "units.json").write_text(json.dumps(registry, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
    wells[["sid", "x", "y", "cell", "layer", "county"]].to_csv(d / "wells.csv", index=False)
    bm[["sid", "x", "y", "cell", "county", "last_survey"]].to_csv(d / "leveling_benchmarks.csv",
                                                                   index=False)
    val = {"checks": checks, "passed": passed, "cross_check_physics_pilot": xchk,
           "overlaps_fresh_vs_design_time": overlaps, "leveling_5fold_alternative": lev_alt,
           "design_time_look_seed": {"time": seed_time, "evidence": seed_src},
           "seconds": round(time.time() - t_start, 1)}
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8")
    here = Path(__file__).resolve().parent
    m = manifest.build_manifest(
        d, product="validation_units", layer="L2",
        inputs=[TWIN_GRID_NPZ, POLYGON, COUNTIES, STATIONS, LEVELING, PILOT_INPUTS],
        input_trees=[WELLS_DIR],
        code=[here / "build_units.py", here / "units.py", here / "__init__.py"],
        parameters={"well_window": [T0, "2022-12"], "leveling_window": [T0, "2022-12"],
                    "kmeans_seed": units.KMEANS_SEED, "fold_seed": units.FOLD_SEED,
                    "fold_diameter_km": 10.0, "buffer_km": {"heads": 5.0, "leveling": 10.0},
                    "river_band_km": 10.0, "proximal_x_km": units.PROXIMAL_X_KM,
                    "station_columns_read": STN_COLS, "sklearn": sklearn.__version__},
        fit_window_end="2022-12", validation_report="validation.json",
        gate={"passed": passed, "criteria": "plan WP A1: unit counts reproduce the plan's "
                                            "design-time numbers, fresh partitions disjoint, "
                                            "no fresh unit duplicates a design-time region, "
                                            "folds keep sites whole and respect buffers"},
        allowed_uses=["selection", "validator", "after_the_fact_analysis"],
        forbidden_uses=["model_design", "prior", "prior_width_input", "calibration_target",
                        "datum_fit", "baseline", "initial_conditions", "idw_field",
                        "forcing_frozen", "scoring_target"],
        personal_data="none: public monitoring-station and benchmark coordinates only; no "
                      "names, addresses or contact fields were read")
    manifest.write_manifest(d, m)
    print(f"wrote {rel_to_repo(d)}; gate passed={passed}", flush=True)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"], "->", c["measured"])
    if not passed:
        return 1

    # ---- seed design-time looks and declare the fresh units (before any downstream test)
    looks = PREREG_DIR / "looks.jsonl"
    already = {lk.get("note", "").split(" (dated to")[0] for lk in prereg.read_chained(looks)}
    dt_units = sorted(u["id"] for u in U if u["role"] == "design_time")
    note = (f"design-time units of the 2026-10-12 pilots (validation_units/{d.name}); seeded "
            "by WP A1 so that they can never serve as fresh validation units")
    if note not in already:
        prereg.record_look(dt_units, "*", "A1", "design_time_seed", seeded_time=seed_time,
                           prereg_dir=PREREG_DIR,
                           note=note + f" (dated to the mtime of _pilots/{seed_src})")
    fresh = sorted(u["id"] for u in U if u["role"] == "fresh")
    doc = {
        "id": f"A1-fresh-units-{d.name}", "wp": "A1", "kind": "declaration",
        "title": "Fresh validation units, folds and coverage gate rule (plan WP A1)",
        "tests": prereg.not_applicable("declaration of units; every downstream test that "
                                       "scores on them (H2, H3, P3, P4m, S1, S5, I1, ...) "
                                       "writes its own pre-registration naming them"),
        "zones": {"design_time": ["dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3"],
                  "note": "Z1-Z3 are design-time choices (plan 4.5); the Yunlin mirror of the "
                          "Z2 rule is fresh.river_band_YL10"},
        "priors": prereg.not_applicable("no model is fitted by A1"),
        "burn_in": prereg.not_applicable("no model is fitted by A1"),
        "folds": {k: by[f"fresh.folds.{k}"]["params"] for k in folds_out},
        "fresh_units": fresh,
        "thresholds": {"coverage_gate": {"target": coverage.TARGET,
                                         "interval": "cluster-adjusted Wilson, ANOVA ICC",
                                         "level": coverage.GATE_LEVEL,
                                         "quantile": coverage.DEFAULT_SMALL_K + " (k-1 df)",
                                         "point_window": list(coverage.POINT_WINDOW),
                                         "point_window_if_n_eff_at_least":
                                             coverage.N_EFF_WINDOW_MIN,
                                         "evidence": "A1_evidence/coverage_sim.json "
                                                     "(design-time, chose the quantile); "
                                                     "A1_evidence/coverage_confirm.json "
                                                     "(pre-registered A1.coverage_confirm.v1)"}},
        "gate_scores": prereg.not_applicable("set by each test's pre-registration"),
        "primary_model": prereg.not_applicable("set by each test's pre-registration"),
        "inputs": [str(d / "units.json"), str(d / "MANIFEST.json"),
                   str(d / "validation.json")],
        "interpretation": (
            "Fresh units (role 'fresh' in units.json) are validation units: no design choice "
            "may be made on them, and every test scored on them is pre-registered first with "
            "the A1 tool (harness.prereg). The two-tercile region is fresh.two_tercile.ME "
            "(63 of 84 Yunlin wells); fresh.two_tercile.WM (55) is secondary. The overlapping "
            "terciles, halves and k-means sets give no independent estimate of tau; report "
            "the effective n. Design-time units (role 'design_time') are labelled as such "
            "wherever they appear and never validate the choices made on them."),
    }
    path, sha = prereg.write_prereg(doc, units_registry=d / "units.json")
    print(f"declared {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())

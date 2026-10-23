"""Per-cell LUIMAP land-use class areas, covariates, county and MOI-township shares, rebuilt
from L0 (port of the landuse_crop_priors pilot's features.py and township.py; logic unchanged,
paths patched), plus whole-township LUIMAP class areas for the in-fan farmland fraction of
plan P3 step 3.

Outputs (WORK = results/dataprep/reports/P3_evidence/work):
  classified_w7.npy    LUIMAP class id per image pixel (cache)
  features.npz         per active cell (pilot layout)
  township.npz         per MOI township x cell shares, in-fan class areas (pilot layout) plus
                       whole_class_area_m2 / whole_points_outside_box for whole townships
Usage: python -I features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

# finest-level codes of the NLSC 109 colour table (the pilot's PALETTE, unchanged)
PALETTE = [
    ("010101", "paddy", (171, 220, 97)), ("010102", "dry", (198, 230, 150)),
    ("010103", "orchard", (99, 192, 59)), ("010200", "aqua", (138, 255, 218)),
    ("010301", "livestock", (112, 134, 79)), ("010302", "livestock", (147, 203, 62)),
    ("010401", "agfac", (107, 144, 75)), ("010402", "agfac", (77, 101, 57)),
    ("020100", "forest", (58, 122, 42)), ("020200", "forest", (56, 168, 0)),
    ("020300", "forest", (79, 190, 15)), ("020401", "forest", (44, 123, 10)),
    ("020402", "forest", (74, 162, 37)), ("020403", "forest", (36, 97, 10)),
    ("020404", "forest", (35, 81, 16)), ("020500", "forest", (122, 186, 101)),
    ("020600", "forest", (4, 108, 31)), ("020700", "forest", (58, 81, 23)),
    ("030100", "transport", (174, 120, 255)), ("030201", "transport", (233, 119, 210)),
    ("030202", "transport", (176, 87, 158)), ("030301", "transport", (130, 81, 212)),
    ("030302", "transport", (99, 62, 162)), ("030401", "transport", (148, 78, 239)),
    ("030402", "transport", (114, 59, 189)), ("030501", "transport", (230, 0, 0)),
    ("030502", "transport", (255, 120, 120)), ("030503", "transport", (255, 2, 165)),
    ("030504", "transport", (234, 148, 163)), ("030505", "transport", (95, 76, 119)),
    ("030601", "transport", (79, 108, 203)), ("030602", "transport", (116, 134, 192)),
    ("030603", "transport", (90, 109, 169)), ("030604", "transport", (63, 72, 101)),
    ("040101", "river", (164, 215, 245)), ("040102", "river", (16, 215, 245)),
    ("040103", "river", (110, 199, 250)), ("040104", "canal", (166, 205, 213)),
    ("040201", "pond", (70, 212, 206)), ("040202", "pond", (121, 238, 233)),
    ("040203", "pond", (122, 216, 250)), ("040300", "river", (215, 136, 0)),
    ("040401", "hydrofac", (156, 156, 156)), ("040402", "hydrofac", (87, 103, 123)),
    ("040403", "hydrofac", (112, 144, 161)), ("040404", "hydrofac", (90, 133, 158)),
    ("040405", "hydrofac", (36, 120, 168)), ("040406", "hydrofac", (92, 125, 145)),
    ("040500", "hydrofac", (204, 204, 204)), ("040600", "sea", (158, 202, 255)),
    ("050101", "commerce", (215, 5, 61)), ("050102", "commerce", (228, 62, 106)),
    ("050200", "resid", (255, 244, 41)), ("050301", "resid", (202, 178, 26)),
    ("050302", "resid", (228, 180, 25)), ("050303", "resid", (236, 193, 83)),
    ("050400", "industry", (255, 194, 61)), ("050500", "industry", (165, 130, 68)),
    ("050600", "public", (197, 0, 255)), ("050700", "public", (131, 131, 131)),
    ("050801", "public", (209, 136, 9)), ("050802", "public", (186, 104, 73)),
    ("060100", "public", (255, 159, 41)), ("060201", "public", (235, 159, 255)),
    ("060202", "public", (227, 158, 255)), ("060203", "public", (218, 130, 255)),
    ("060204", "public", (209, 101, 235)), ("060205", "public", (201, 73, 255)),
    ("060300", "public", (161, 139, 246)), ("060400", "public", (249, 224, 165)),
    ("060501", "public", (148, 172, 176)), ("060502", "public", (147, 161, 232)),
    ("060503", "public", (239, 113, 135)), ("060504", "public", (164, 175, 232)),
    ("060505", "public", (122, 154, 148)), ("060600", "public", (216, 142, 103)),
    ("070101", "recreation", (205, 205, 93)), ("070102", "recreation", (233, 233, 133)),
    ("070103", "recreation", (230, 230, 147)), ("070200", "recreation", (147, 255, 47)),
    ("070301", "recreation", (246, 171, 171)), ("070302", "recreation", (172, 172, 247)),
    ("080100", "mining", (206, 154, 29)), ("080200", "mining", (149, 111, 35)),
    ("080300", "mining", (141, 141, 141)), ("090100", "grass", (0, 168, 133)),
    ("090200", "grass", (176, 211, 38)), ("090301", "grass", (201, 173, 137)),
    ("090302", "grass", (154, 137, 105)), ("090303", "grass", (192, 108, 34)),
    ("090400", "unused", (51, 51, 51)), ("090501", "unused", (192, 174, 84)),
    ("090502", "unused", (188, 146, 62)),
    ("nodata_k", "nodata", (0, 0, 0)), ("nodata_w", "nodata", (255, 255, 255)),
]
GROUPS = ["paddy", "dry", "orchard", "aqua", "livestock", "agfac", "forest", "transport",
          "river", "canal", "pond", "hydrofac", "sea", "commerce", "resid", "industry",
          "public", "recreation", "mining", "grass", "unused", "nodata", "unk"]


def classify(rgb: np.ndarray, tol: float = 8.0) -> np.ndarray:
    pal = np.array([p[2] for p in PALETTE], dtype=np.int32)
    gid = np.array([GROUPS.index(p[1]) for p in PALETTE])
    out = np.full(len(rgb), GROUPS.index("unk"), dtype=np.int16)
    for i0 in range(0, len(rgb), 2_000_000):
        r = rgb[i0:i0 + 2_000_000].astype(np.int32)
        d2 = ((r[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        ok = d2[np.arange(len(r)), j] <= tol * tol
        o = out[i0:i0 + 2_000_000]
        o[ok] = gid[j[ok]]
    return out


def classified_image(img: np.ndarray, win: int = 7) -> np.ndarray:
    """Class id per image pixel; blended (anti-aliased) pixels take the majority known class of
    their ``win`` x ``win`` neighbourhood (the pilot's rule). Cached in WORK."""
    from scipy import ndimage

    cache = lib.WORK / f"classified_w{win}.npy"
    if cache.exists():
        return np.load(cache)
    code = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2]
    u, inv = np.unique(code.ravel(), return_inverse=True)
    urgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1)
    ucls = classify(urgb)
    cls = ucls[inv].reshape(code.shape).astype(np.int8)
    unk = GROUPS.index("unk")
    print("pixels unk before fill:", float((cls == unk).mean()), flush=True)
    best = np.zeros(cls.shape, np.float32)
    arg = np.full(cls.shape, unk, np.int8)
    for k in range(len(GROUPS)):
        if k == unk or GROUPS[k] == "nodata":
            continue
        cnt = ndimage.uniform_filter((cls == k).astype(np.float32), size=win, mode="nearest")
        upd = cnt > best
        best[upd] = cnt[upd]
        arg[upd] = k
        del cnt, upd
    fill = (cls == unk) & (best > 0)
    cls[fill] = arg[fill]
    print("pixels unk after fill:", float((cls == unk).mean()), flush=True)
    lib.WORK.mkdir(parents=True, exist_ok=True)
    np.save(cache, cls)
    return cls


class LuimapSampler:
    def __init__(self):
        from PIL import Image
        from pyproj import Transformer

        Image.MAX_IMAGE_PIXELS = None
        img = np.asarray(Image.open(lib.LUIMAP_PNG).convert("RGB"))
        self.H, self.W = img.shape[:2]
        self.cls = classified_image(img)
        del img
        self.west, self.south, self.east, self.north = np.load(lib.LUIMAP_BOX)
        self.tr = Transformer.from_crs(3826, 4326, always_xy=True)

    @staticmethod
    def _merc_y(a):
        return np.log(np.tan(np.pi / 4 + np.radians(a) / 2))

    def sample(self, X, Y, with_inside: bool = False):
        lon, lat = self.tr.transform(X, Y)
        fx = (lon - self.west) / (self.east - self.west) * self.W
        fy = ((self._merc_y(self.north) - self._merc_y(lat))
              / (self._merc_y(self.north) - self._merc_y(self.south)) * self.H)
        px = np.clip(fx.astype(np.int64), 0, self.W - 1)
        py = np.clip(fy.astype(np.int64), 0, self.H - 1)
        g = self.cls[py, px]
        if with_inside:
            inside = (fx >= 0) & (fx < self.W) & (fy >= 0) & (fy < self.H)
            return g, inside
        return g


def lattice(grid):
    xs = grid.x0 + (np.arange(int(grid.nx * grid.dx / lib.STEP)) + 0.5) * lib.STEP
    ys = grid.y0 + (np.arange(int(grid.ny * grid.dx / lib.STEP)) + 0.5) * lib.STEP
    XX, YY = np.meshgrid(xs, ys)
    X, Y = XX.ravel(), YY.ravel()
    col = ((X - grid.x0) // grid.dx).astype(np.int64)
    row = ((Y - grid.y0) // grid.dx).astype(np.int64)
    inside = grid.mask[row, col]
    active = -np.ones((grid.ny, grid.nx), dtype=np.int64)
    r_, c_ = np.nonzero(grid.mask)
    active[r_, c_] = np.arange(r_.size)
    cell = active[row, col]
    return X[inside], Y[inside], cell[inside]


def build_features(grid, lu: LuimapSampler, X, Y, cell, g):
    import geopandas as gpd
    import shapely
    from scipy import ndimage

    a_pt = lib.STEP * lib.STEP
    A = grid.n_active
    area = np.zeros((len(GROUPS), A))
    for k in range(len(GROUPS)):
        s = g == k
        area[k] = np.bincount(cell[s], minlength=A) * a_pt
    cty = gpd.read_file(lib.TW_COUNTY).to_crs(3826)
    polys = {"changhua": cty[cty.NAME_2 == "Changhua"].geometry.union_all(),
             "yunlin": cty[cty.NAME_2 == "Yulin"].geometry.union_all(),
             "nantou": cty[cty.NAME_2 == "Nantou"].geometry.union_all()}
    cid = np.full(len(X), 3, dtype=np.int8)
    for i, p in enumerate(polys.values()):
        cid[shapely.contains_xy(p, X, Y)] = i
    share = np.zeros((4, A))
    for i in range(4):
        share[i] = np.bincount(cell[cid == i], minlength=A) * a_pt / grid.dx ** 2
    area_cty = np.zeros((4, len(GROUPS), A))
    cty_area = np.zeros((4, len(GROUPS)))
    for i in range(4):
        for k in range(len(GROUPS)):
            s_ = (cid == i) & (g == k)
            area_cty[i, k] = np.bincount(cell[s_], minlength=A) * a_pt
            cty_area[i, k] = s_.sum() * a_pt
    border = polys["changhua"].boundary.intersection(polys["yunlin"].buffer(50.0))
    sea = gpd.read_file(lib.SEA_SHP).geometry.iloc[0]
    r_, c_ = np.nonzero(grid.mask)
    cxy = np.stack([grid.x0 + (c_ + 0.5) * grid.dx, grid.y0 + (r_ + 0.5) * grid.dx], 1)
    pts = shapely.points(cxy[:, 0], cxy[:, 1])
    d_river = shapely.distance(border, pts)
    d_coast = shapely.distance(sea, pts)
    side = np.where(share[0] >= share[1], 1.0, -1.0)
    dem = np.load(lib.BASEMAP)["dem"].astype(float)
    canal2d = np.zeros((grid.ny, grid.nx))
    canal2d[r_, c_] = area[GROUPS.index("canal")]
    canal3 = ndimage.uniform_filter(canal2d, size=3, mode="constant")[r_, c_]
    np.savez_compressed(
        lib.WORK / "features.npz", groups=np.array(GROUPS), area_m2=area,
        county_names=np.array(["changhua", "yunlin", "nantou", "other"]), county_share=share,
        county_class_area_m2=cty_area, area_cty_m2=area_cty, cxy=cxy, rc=np.stack([r_, c_], 1),
        d_river_m=d_river, d_river_signed_m=d_river * side, d_coast_m=d_coast, dem=dem,
        canal3_m2=canal3, step_m=lib.STEP)
    print("features.npz written", flush=True)


def build_township(grid, lu: LuimapSampler, X, Y, cell, g):
    import geopandas as gpd
    import shapely

    tw = gpd.read_file(lib.TOWNS_TOPO, layer="towns")
    tw = tw.set_crs(4326, allow_override=True).to_crs(3826)
    tw = tw[tw.COUNTYNAME.isin(["彰化縣", "雲林縣", "南投縣", "嘉義縣"])].reset_index(drop=True)
    names = [f"{c}{t}" for c, t in zip(tw.COUNTYNAME, tw.TOWNNAME, strict=True)]
    tid = np.full(len(X), -1, dtype=np.int32)
    for i, geom in enumerate(tw.geometry):
        tid[shapely.contains_xy(geom, X, Y)] = i
    A = grid.n_active
    nT = len(names)
    a_pt = lib.STEP * lib.STEP
    share = np.zeros((nT, A))
    for i in range(nT):
        share[i] = np.bincount(cell[tid == i], minlength=A) * a_pt / grid.dx ** 2
    t_cls = np.zeros((nT, len(GROUPS)))
    for i in range(nT):
        s = tid == i
        t_cls[i] = np.bincount(g[s], minlength=len(GROUPS))[:len(GROUPS)] * a_pt
    maj = np.where(share.max(0) > 0, share.argmax(0), -1)
    # whole-township LUIMAP class areas (new in P3; Changhua and Yunlin townships only)
    whole = np.full((nT, len(GROUPS)), np.nan)
    outside = np.full(nT, np.nan)
    for i, geom in enumerate(tw.geometry):
        if not names[i].startswith(("彰化縣", "雲林縣")):
            continue
        x0, y0, x1, y1 = geom.bounds
        xs = np.arange(np.floor(x0 / lib.STEP) * lib.STEP + lib.STEP / 2, x1, lib.STEP)
        ys = np.arange(np.floor(y0 / lib.STEP) * lib.STEP + lib.STEP / 2, y1, lib.STEP)
        XX, YY = np.meshgrid(xs, ys)
        xx, yy = XX.ravel(), YY.ravel()
        ins = shapely.contains_xy(geom, xx, yy)
        gg, inbox = lu.sample(xx[ins], yy[ins], with_inside=True)
        whole[i] = np.bincount(gg[inbox], minlength=len(GROUPS))[:len(GROUPS)] * a_pt
        outside[i] = float((~inbox).sum()) * a_pt
    np.savez_compressed(lib.WORK / "township.npz", names=np.array(names), share=share,
                        cell_town=maj, town_class_area_m2=t_cls,
                        town_area_total_m2=tw.geometry.area.to_numpy(),
                        groups=np.array(GROUPS), whole_class_area_m2=whole,
                        whole_points_outside_box_m2=outside)
    print("township.npz written; unassigned lattice fraction", float((tid < 0).mean()),
          flush=True)


def main():
    from common import hashing

    lib.WORK.mkdir(parents=True, exist_ok=True)
    grid = load = lib.load_grid()
    del load
    c0 = np.load(lib.C0_PRE / "cells_by_class.npz", allow_pickle=True)
    mh = hashing.mask_hash(grid.mask)
    assert mh == str(c0["mask_hash"]), (mh, str(c0["mask_hash"]))
    assert (grid.nx, grid.ny, grid.dx) == (int(c0["nx"]), int(c0["ny"]), float(c0["dx"]))
    assert abs(grid.x0 - float(c0["x0"])) < 1e-6 and abs(grid.y0 - float(c0["y0"])) < 1e-6
    print("grid matches C0 preorigin/v3 (mask hash", mh[:12], ")", flush=True)
    lu = LuimapSampler()
    X, Y, cell = lattice(grid)
    g = lu.sample(X, Y)
    print("lattice points in fan:", len(g), flush=True)
    build_features(grid, lu, X, Y, cell, g)
    build_township(grid, lu, X, Y, cell, g)


if __name__ == "__main__":
    main()

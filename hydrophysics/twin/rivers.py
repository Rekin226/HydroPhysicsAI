"""River cells for the flow model: the Choushui, Wu and Beigang as head-dependent boundaries.

Until 2026-09-23 every river on the fan was a no-flow line: the solver saw the Choushui,
which crosses the whole fan and loses water to the proximal gravels, only as more of the
same aquifer. This module turns the channel polygons in the local river layer into a set of
river cells the solver can exchange water with (``FlowModel.set_rivers``):

- **Where.** ``chou-shui-data/data/water/river_TWD97.shp`` holds channel *polygons*
  (13,294 features, names in Big5, no ``.prj``: it is EPSG:3826). A cell's weight is the
  channel area inside it over the cell area, from an STR-tree of cell boxes against the
  dissolved group polygon; cells under 2 % are dropped. The Wu and the Beigang run along
  the fan's northern and southern edges, mostly outside the mask, so for those "edge"
  groups every active cell whose box, grown by ``edge_buffer_m``, touches the channel gets
  weight 1: one exposed face, like the coast.
- **Stage.** ``h_riv = dem - stage_depth`` with the SRTM elevation the basemap fetched
  (``results/twin/basemap.npz["dem"]``, one value per active cell), then hydro-flattened
  per group: ordered from the apex (east) to the sea (west), a running minimum, so the
  stage never rises downstream. River bottom ``rbot = dem - rbot_depth``.
- **Physics.** ``ghb``: flux ``C w (h_riv - h)``, linear. ``riv`` (MODFLOW RIV): the same
  while the aquifer head is above ``rbot``; once it falls below, the leakage is the
  constant ``C w (h_riv - rbot)``. The switch is lagged one step and detached. That regime
  matters here: the proximal Choushui is a losing river whose aquifer heads sit far below
  its bed.

Options (2026-09-23, second round, each off by default):

- ``c_split="zone"`` (``--river-c-split zone``): one conductance per fan zone for every
  non-edge group (``choushui_proximal``, ``choushui_mid``, ...). The proximal Choushui
  loses water to the gravels and the distal reach may gain it; one conductance cannot do
  both.
- ``exclude_idx`` (``--river-skip-apex``): drop river cells that are also apex GHB cells,
  so the Choushui's entry is not counted twice (once as the apex inflow boundary, once as
  river leakage).
- ``season`` (``--river-stage-season CSV``): a monthly connection factor per group that
  multiplies the conductance (the Jiji weir diverts most dry-season flow, so the bed is
  often dry). ``load_river_season`` reads it; without one the factor is 1.

The fan polygon in the ``.shp`` layers is in TWD67 (EPSG:3828) and sits ~0.8 km off TWD97;
the model grid comes from the GeoJSON reprojected to 3826, so rivers are always matched
against the grid, never against that polygon.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

# Channel names as they appear in river_TWD97.shp (Big5-decoded).
RIVER_GROUPS: dict[str, tuple[str, ...]] = {
    "choushui": ("濁水溪(西螺溪)", "濁水溪"),
    "wu": ("烏溪(大肚溪)",),
    "beigang": ("北港溪",),
    # drainage channels; kept out of the default set
    "minor": ("新虎尾溪", "舊濁水溪(麥嶼溪)", "虎尾溪"),
}
EDGE_GROUPS = ("wu", "beigang")
DEFAULT_RIVER_SET = "choushui,wu,beigang"
MIN_WEIGHT = 0.02


def default_river_shp() -> str:
    base = os.environ.get("HYDROMIND_GW_DATA", os.path.join("chou-shui-data", "data"))
    return os.path.join(base, "water", "river_TWD97.shp")


@dataclass(frozen=True)
class RiverSet:
    """River cells, one entry per (cell, group); a cell can appear in two groups."""

    idx: np.ndarray        # (n,) active-cell index
    weight: np.ndarray     # (n,) channel area / cell area, or 1 for an edge group
    group: np.ndarray      # (n,) index into ``names``
    h_riv: np.ndarray      # (n,) river stage, m
    rbot: np.ndarray       # (n,) river-bed bottom, m
    names: tuple[str, ...]
    # (n_groups, 12) monthly connection factor (January first), or None = always 1
    season: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return int(len(self.idx))

    def describe(self) -> str:
        parts = []
        for g, name in enumerate(self.names):
            sel = self.group == g
            parts.append(f"{name} {int(sel.sum())} cells (sum w {self.weight[sel].sum():.1f}, "
                         f"stage {self.h_riv[sel].min():.1f}..{self.h_riv[sel].max():.1f} m)"
                         if sel.any() else f"{name} 0 cells")
        return "; ".join(parts)


def parse_river_set(text: str) -> tuple[str, ...]:
    names = tuple(s.strip() for s in str(text).split(",") if s.strip())
    bad = [n for n in names if n not in RIVER_GROUPS]
    if bad or not names:
        raise ValueError(f"--river-set: unknown group(s) {bad}; expected from "
                         f"{sorted(RIVER_GROUPS)}")
    return names


def load_river_polygons(path: str, groups: tuple[str, ...]) -> dict:
    """``{group: dissolved shapely geometry}`` in EPSG:3826 from the channel layer."""
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(path, encoding="big5")
    out = {}
    for g in groups:
        sel = gdf[gdf["NAME"].isin(RIVER_GROUPS[g])]
        if sel.empty:
            raise ValueError(f"river group {g!r}: none of {RIVER_GROUPS[g]} in {path}")
        out[g] = unary_union(list(sel.geometry.buffer(0)))
    return out


def _flatten_stage(idx: np.ndarray, h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Running minimum from east (apex) to west (sea): stage never rises downstream."""
    order = np.argsort(-x, kind="stable")
    out = h.copy()
    out[order] = np.minimum.accumulate(h[order])
    return out


def river_cells(grid, polys: dict, dem: np.ndarray, stage_depth: float = 1.0,
                rbot_depth: float = 3.0, edge_buffer_m: float | None = None,
                min_weight: float = MIN_WEIGHT, zone_of_cell: np.ndarray | None = None,
                c_split: str = "group", exclude_idx: np.ndarray | None = None
                ) -> RiverSet:
    """Intersect the group polygons with the grid -> ``RiverSet`` (see the module doc).

    ``c_split="zone"`` splits every non-edge group by ``zone_of_cell`` into one group
    per zone present (``name_zone``); ``exclude_idx`` drops those cells (the apex)."""
    if c_split not in ("group", "zone"):
        raise ValueError(f"c_split must be 'group' or 'zone', got {c_split!r}")
    if c_split == "zone" and zone_of_cell is None:
        raise ValueError("--river-c-split zone needs the zone assignment (--param-mode zonal)")
    from shapely import STRtree
    from shapely.geometry import box

    dem = np.asarray(dem, dtype="float64").reshape(-1)
    if dem.shape != (grid.n_active,):
        raise ValueError(f"dem has {dem.shape[0]} values but the grid has {grid.n_active} "
                         "active cells -- re-run hydrophysics.twin.basemap at this --dx")
    if rbot_depth < stage_depth:
        raise ValueError("rbot_depth must be at least stage_depth (bottom below stage)")
    buf = float(grid.dx if edge_buffer_m is None else edge_buffer_m)
    cent = grid.centroids()
    half = grid.dx / 2.0
    boxes = [box(x - half, y - half, x + half, y + half) for x, y in cent]
    tree = STRtree(boxes)
    area = grid.dx ** 2
    ids, W, G, H, R = [], [], [], [], []
    names = tuple(polys)
    for gi, name in enumerate(names):
        poly = polys[name]
        if name in EDGE_GROUPS:
            hits = np.asarray(tree.query(poly.buffer(buf), predicate="intersects"))
            cells = np.unique(hits).astype("int64")
            w = np.ones(len(cells))
        else:
            hits = np.unique(np.asarray(tree.query(poly, predicate="intersects")))
            w = np.array([boxes[i].intersection(poly).area / area for i in hits])
            keep = w >= min_weight
            cells, w = hits[keep].astype("int64"), w[keep]
        if len(cells) == 0:
            continue
        h = dem[cells] - float(stage_depth)
        h = _flatten_stage(cells, h, cent[cells, 0])
        ids.append(cells)
        W.append(w)
        G.append(np.full(len(cells), gi, dtype="int64"))
        H.append(h)
        R.append(h - (float(rbot_depth) - float(stage_depth)))
    if not ids:
        raise ValueError("no river cell intersects the grid")
    idx, grp = np.concatenate(ids), np.concatenate(G)
    W, H, R = np.concatenate(W), np.concatenate(H), np.concatenate(R)
    if exclude_idx is not None and len(exclude_idx):
        keep = ~np.isin(idx, np.asarray(exclude_idx, dtype="int64"))
        idx, grp, W, H, R = idx[keep], grp[keep], W[keep], H[keep], R[keep]
    if c_split == "zone":
        from .zones import ZONE_NAMES

        zoc = np.asarray(zone_of_cell, dtype="int64").reshape(-1)
        new_names: list[str] = []
        new_grp = np.empty_like(grp)
        for gi, name in enumerate(names):
            sel = grp == gi
            if name in EDGE_GROUPS:
                new_grp[sel] = len(new_names)
                new_names.append(name)
                continue
            for zi, zname in enumerate(ZONE_NAMES):
                sz = sel & (zoc[idx] == zi)
                if sz.any():
                    new_grp[sz] = len(new_names)
                    new_names.append(f"{name}_{zname}")
        grp, names = new_grp, tuple(new_names)
    if len(idx) == 0:
        raise ValueError("no river cell left after excluding the apex cells")
    return RiverSet(idx=idx, weight=W, group=grp, h_riv=H, rbot=R, names=names)


def load_river_season(path: str, names: tuple[str, ...]) -> np.ndarray:
    """``--river-stage-season CSV`` -> ``(n_groups, 12)`` connection factors in [0, 1].

    The CSV has a ``month`` column (1-12) and one column per group; a group of a zone
    split (``choushui_mid``) falls back to its river's column (``choushui``), and a group
    with no column at all gets 1 (always connected)."""
    import pandas as pd

    df = pd.read_csv(path)
    if "month" not in df.columns or sorted(df["month"].astype(int)) != list(range(1, 13)):
        raise ValueError(f"{path}: needs a 'month' column with the 12 months 1..12")
    df = df.set_index(df["month"].astype(int)).sort_index()
    out = np.ones((len(names), 12))
    for g, name in enumerate(names):
        col = name if name in df.columns else name.split("_")[0]
        if col in df.columns:
            out[g] = df[col].astype(float).to_numpy()
    if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise ValueError(f"{path}: connection factors must be finite and in [0, 1]")
    return out


def apex_overlap(rs: RiverSet, apex_idx: np.ndarray) -> int:
    """How many river cells are also apex boundary cells (should be 0 with skip-apex)."""
    return int(np.isin(rs.idx, np.asarray(apex_idx)).sum())


def load_dem(dem_npz: str, grid) -> tuple[np.ndarray, str]:
    """``(dem (A,), sha1 of the array)`` from the basemap cache, checked against the grid."""
    if not os.path.exists(dem_npz):
        raise FileNotFoundError(f"{dem_npz} not found -- run python -m "
                                "hydrophysics.twin.basemap first")
    with np.load(dem_npz) as z:
        if "dem" not in z:
            raise ValueError(f"{dem_npz} has no 'dem' (it was fetched with --no-dem)")
        dem = np.asarray(z["dem"], dtype="float64")
    if dem.shape != (grid.n_active,):
        raise ValueError(f"{dem_npz}: dem has {dem.shape[0]} cells, grid has "
                         f"{grid.n_active}; re-run basemap at dx={grid.dx:g}")
    return dem, hashlib.sha1(np.ascontiguousarray(dem).tobytes()).hexdigest()


_CACHE: dict = {}


def build_river_set(grid, shp: str | None = None, groups: str = DEFAULT_RIVER_SET,
                    dem_npz: str = "results/twin/basemap.npz", stage_depth: float = 1.0,
                    rbot_depth: float = 3.0, edge_buffer_m: float | None = None,
                    zone_of_cell: np.ndarray | None = None, c_split: str = "group",
                    exclude_idx: np.ndarray | None = None, season_csv: str | None = None
                    ) -> tuple[RiverSet, str]:
    """Everything above in one call -> ``(RiverSet, dem_sha1)``; memoised per argument set
    so an ensemble of members rebuilds it once. The defaults of the second-round options
    (``c_split``, ``exclude_idx``, ``season_csv``) build the historical set."""
    import dataclasses

    shp = shp or default_river_shp()
    zkey = (None if zone_of_cell is None or c_split == "group"
            else hashlib.sha1(np.ascontiguousarray(zone_of_cell, dtype="int64")).hexdigest())
    xkey = (None if exclude_idx is None
            else tuple(int(i) for i in np.asarray(exclude_idx).ravel()))
    key = (shp, groups, dem_npz, float(stage_depth), float(rbot_depth), edge_buffer_m,
           float(grid.dx), grid.n_active, float(grid.x0), float(grid.y0), c_split, zkey,
           xkey, season_csv)
    if key not in _CACHE:
        dem, sha = load_dem(dem_npz, grid)
        polys = load_river_polygons(shp, parse_river_set(groups))
        rs = river_cells(grid, polys, dem, stage_depth=stage_depth, rbot_depth=rbot_depth,
                         edge_buffer_m=edge_buffer_m, zone_of_cell=zone_of_cell,
                         c_split=c_split, exclude_idx=exclude_idx)
        if season_csv:
            rs = dataclasses.replace(rs, season=load_river_season(season_csv, rs.names))
        _CACHE[key] = (rs, sha)
    return _CACHE[key]

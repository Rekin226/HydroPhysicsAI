"""The per-cell county map behind ``--zone-county-split`` (ladder rung R1c, 2026-10-15).

Rung R1 added the Changhua pump census to the calibration and failed its pre-registration:
Changhua draws about as much pump electricity as Yunlin, but its heads swing about a third
as much and it subsides about a quarter as much, while every hydraulic and column parameter
is zoned by easting only (``zones.fan_zones``). R1c asks whether Changhua's aquifer and
column respond differently: every zone gets a Changhua copy of its parameters. This module
supplies the per-cell county flag and the zone maps every consumer rebuilds from a
calibration's meta.

Rule (map v2, 2026-10-17): a cell is Changhua when its centroid lies in (or on) the
彰化縣 polygon of ``TW_COUNTY.gpkg``; a centroid inside no county polygon (the coast, where
the generalised GADM coastline runs inland of the fan polygon) takes the county of the
NEAREST polygon. Every other cell is "rest": Yunlin and the Nantou and Chiayi fringes. The
gpkg's coordinates are TWD97 / TM2 metres already: the file is GADM (natively WGS84)
projected to TM2 and mis-tagged EPSG:3828 (TWD67). Reprojecting it with ``to_crs(3826)``
(map v1) applied a spurious TWD67 -> TWD97 datum shift of about +829 m E / -207 m N, which
moved the county line and the coastline, mislabelled 42 cells and left the coastal 芳苑
nest (07230312/22/32, three of the 74 scored Changhua wells) and 28 leveling benchmarks on
the Yunlin side; ``_counties`` therefore overrides the tag (``set_crs(3826,
allow_override=True)``). The tag is checked against the independent TPC pump census,
whose county comes from each pump's address: v2 labels 99.996 % of the pump horsepower
with its own county (v1 99.2 %).

Riverbed cells on the Choushui that straddle the county line are labelled by their
centroid like every other cell; on this grid the centroid rule agrees with the area
majority of every land cell (``validation.json``: ``area_majority_disagreements``), and
the straddling channel cells hold almost no pumps -- their parameters are set by the
river's hydraulics more than by the county, so their label is a convention, recorded here.

The build gates the map on independent labels (``validation.json``): (a) the pump census
-- >= 99 % of pump horsepower in a cell of its own county, and a cell whose horsepower
majority disagrees with its flag must lie mostly in its flagged county with every
disagreeing pump within 500 m of the county line (the GADM line along the braided Choushui
channel is generalised) and all such cells together hold <= 0.1 % of the horsepower (v2:
three riverbed cells, 82-85 % Changhua by area, holding 24.5 HP of Yunlin pumps); (b) every WRA
station of ``fan_stations.parquet`` labelled 彰化縣 lies in a Changhua cell and every one
labelled 雲林縣 does not; (c) the cells inside no county polygon are counted with their
maximum distance to land (<= 1 km).

Product: ``results/dataprep/zone_county_map/v2/zone_county_map.npz`` with a
``MANIFEST.json`` written by the dataprep staging library, built by

    python -m hydrophysics.twin.zone_county build

``county_sha256`` is the sha256 of the per-cell flag as uint8 bytes in active-cell order
(the content, not the file). A calibration records it as ``zone_county_sha256``; every
consumer rebuilds the map (from the cached npz when it was built on the same grid, else
from the gpkg) and refuses one whose hash differs. ``calibrate_flow`` additionally
rebuilds the map from the gpkg and requires it to equal the cache before it fits.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys

import numpy as np

from .zones import (
    COUNTY_SUFFIX,
    N_ZONES,
    county_split,
    county_split_weights,
    fan_zones,
    zone_blend_weights,
    zone_labels,
)

COUNTY_GPKG = "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
COUNTY_FIELD = "NL_NAME_2"
CHANGHUA = "彰化縣"
TARGET_EPSG = 3826
PRODUCT = "zone_county_map"
NPZ_NAME = "zone_county_map.npz"
MAP_VERSION = "v2"
DEFAULT_MAP = f"results/dataprep/{PRODUCT}/{MAP_VERSION}/{NPZ_NAME}"
RULE = ("cell centroid in (or on) the 彰化縣 polygon of TW_COUNTY.gpkg (coordinates read as "
        "EPSG:3826, the file's EPSG:3828 tag overridden); a centroid in no county polygon "
        "takes the nearest polygon's county -> Changhua when that is 彰化縣; every other "
        "cell -> rest")
# the CRS the gpkg's coordinates are actually in (its EPSG:3828 tag is wrong; see above)
GPKG_TRUE_EPSG = 3826
PUMP_CENSUS = "results/dataprep/pump_electricity/model_census/v1/census.parquet"
STATIONS = "AMP_V2/data/fan_stations.parquet"
# validation gates of build_product
GATE_HP_AGREEMENT = 0.99
# the GADM line along the braided Choushui channel is generalised by a few hundred metres
GATE_BORDER_M = 500.0
GATE_DISAGREE_HP_MAX = 0.001
GATE_NO_POLYGON_MAX_M = 1000.0
# the theta-meta keys a county-split calibration records (only when the split is on)
META_KEYS = ("zone_county_split", "zone_county_map", "zone_county_sha256",
             "zone_county_npz_sha256", "zone_county_n_changhua")


def _norm(name) -> str:
    return str(name).replace("臺", "台")


@functools.lru_cache(maxsize=4)
def _counties(gpkg: str):
    """The county table in TWD97 / TM2 metres (cached per path). The file's coordinates are
    TM2 already and its EPSG:3828 tag is wrong, so the tag is overridden, never
    reprojected (a reprojection applies a spurious ~850 m datum shift; module docstring)."""
    import geopandas as gpd

    if not os.path.exists(gpkg):
        raise FileNotFoundError(f"county polygons {gpkg!r} not found (the zone county map "
                                "is built from them)")
    cty = gpd.read_file(gpkg).set_crs(GPKG_TRUE_EPSG, allow_override=True)
    if GPKG_TRUE_EPSG != TARGET_EPSG:
        cty = cty.to_crs(TARGET_EPSG)
    if COUNTY_FIELD not in cty.columns:
        raise ValueError(f"{gpkg}: no {COUNTY_FIELD} column ({list(cty.columns)})")
    cty = cty.assign(county=cty[COUNTY_FIELD].map(_norm))
    return cty


def county_of_points(xy: np.ndarray, gpkg: str = COUNTY_GPKG
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 2)`` TWD97 points (metres) -> ``(county, distance_m)``: the county polygon
    holding each point (distance 0; a point on a boundary counts as inside), else the
    nearest polygon's county and its distance."""
    import geopandas as gpd

    cty = _counties(gpkg)
    xy = np.asarray(xy, dtype="float64").reshape(-1, 2)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=TARGET_EPSG)
    nn = gpd.sjoin_nearest(pts, cty[["county", "geometry"]], how="left", distance_col="d_m")
    nn = nn[~nn.index.duplicated(keep="first")].sort_index()
    return nn["county"].to_numpy(dtype=object), nn["d_m"].to_numpy(dtype="float64")


def changhua_flags(xy: np.ndarray, gpkg: str = COUNTY_GPKG) -> np.ndarray:
    """``(n, 2)`` TWD97 points (metres) -> ``(n,)`` bool: inside (or on) the 彰化縣 polygon,
    or inside no county polygon and nearest to it (the module's rule)."""
    import shapely

    cty = _counties(gpkg)
    sel = cty[cty["county"] == _norm(CHANGHUA)]
    if sel.empty:
        raise ValueError(f"{gpkg}: no {CHANGHUA} polygon in {COUNTY_FIELD}")
    xy = np.asarray(xy, dtype="float64").reshape(-1, 2)
    inside = np.asarray(shapely.intersects_xy(sel.union_all(), xy[:, 0], xy[:, 1]), dtype=bool)
    in_any = np.asarray(shapely.intersects_xy(cty.union_all(), xy[:, 0], xy[:, 1]), dtype=bool)
    out = inside.copy()
    none = np.flatnonzero(~in_any)
    if none.size:
        county, _ = county_of_points(xy[none], gpkg)
        out[none] = county == _norm(CHANGHUA)
    return out


def county_sha256(flags: np.ndarray) -> str:
    """sha256 of the per-cell flag as uint8 bytes (active-cell order): the map's identity."""
    b = np.ascontiguousarray(np.asarray(flags).astype(bool).astype("uint8")).tobytes()
    return hashlib.sha256(b).hexdigest()


def _file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _grid_record(grid) -> dict:
    return {"nx": int(grid.nx), "ny": int(grid.ny), "dx": float(grid.dx),
            "x0": float(grid.x0), "y0": float(grid.y0),
            "mask": np.asarray(grid.mask, dtype=bool)}


def _same_grid(z, grid) -> bool:
    try:
        return (int(z["nx"]) == int(grid.nx) and int(z["ny"]) == int(grid.ny)
                and float(z["dx"]) == float(grid.dx) and float(z["x0"]) == float(grid.x0)
                and float(z["y0"]) == float(grid.y0)
                and np.array_equal(np.asarray(z["mask"], dtype=bool),
                                   np.asarray(grid.mask, dtype=bool)))
    except (KeyError, ValueError):
        return False


def county_diagnostics(grid, flags: np.ndarray, gpkg: str = COUNTY_GPKG,
                       zone_boundaries: str = "205,182,208") -> dict:
    """What the map holds: cells per county polygon, the cells inside no county polygon
    with the county they take (the nearest polygon's) and their distance to it, the cells
    straddling the county line and whether the centroid rule agrees with each land cell's
    area majority, and Changhua cells per zone of ``zone_boundaries``."""
    import shapely

    cty = _counties(gpkg)
    c = grid.centroids()
    per_county = {}
    in_any = np.zeros(len(c), dtype=bool)
    for name in sorted(cty["county"].unique()):
        hit = shapely.intersects_xy(cty[cty["county"] == name].union_all(), c[:, 0], c[:, 1])
        in_any |= hit
        if hit.any():
            per_county[name] = int(hit.sum())
    none_idx = np.flatnonzero(~in_any)
    nearest: dict = {}
    max_d = 0.0
    if none_idx.size:
        county, d_m = county_of_points(c[none_idx], gpkg)
        max_d = float(d_m.max())
        for name in sorted(set(county.tolist())):
            sel = county == name
            nearest[name] = {"n_cells": int(sel.sum()), "max_distance_m": float(d_m[sel].max()),
                             "labelled_changhua": bool(name == _norm(CHANGHUA))}
    # the area of each cell in the Changhua polygon, as a share of its land area
    half = float(grid.dx) / 2.0
    boxes = shapely.box(c[:, 0] - half, c[:, 1] - half, c[:, 0] + half, c[:, 1] + half)
    ch_geom = cty[cty["county"] == _norm(CHANGHUA)].union_all()
    land = shapely.area(shapely.intersection(boxes, cty.union_all()))
    ch_area = shapely.area(shapely.intersection(boxes, ch_geom))
    has_land = land > 0.0
    frac = np.where(has_land, ch_area / np.where(has_land, land, 1.0), np.nan)
    straddle = has_land & (frac > 0.0) & (frac < 1.0)
    area_major = frac > 0.5
    from .calibrate_flow import _parse_zone_boundaries

    p, d, s = _parse_zone_boundaries(zone_boundaries, allow_split=True)
    z = fan_zones(c, p, d, s)
    labels = zone_labels(s is not None)
    per_zone = {name: {"cells": int((z == i).sum()), "changhua": int(((z == i) & flags).sum())}
                for i, name in enumerate(labels)}
    return {"n_active": int(grid.n_active), "n_changhua": int(flags.sum()),
            "n_rest": int((~flags).sum()), "cells_per_county_polygon": per_county,
            "cells_in_no_county_polygon": int(none_idx.size),
            "no_county_cells_max_distance_m": max_d,
            "no_county_cells_nearest_county": nearest,
            "cells_straddling_county_line": int(straddle.sum()),
            "area_majority_disagreements": int((area_major != flags)[has_land].sum()),
            "zone_boundaries": zone_boundaries, "per_zone": per_zone}


def independent_label_gates(grid, flags: np.ndarray, gpkg: str = COUNTY_GPKG,
                            census: str = PUMP_CENSUS, stations: str = STATIONS) -> dict:
    """The map against county labels it was NOT built from (``build_product``'s gate):

    (a) the pump census (county from each pump's address): the share of Changhua +
        Yunlin pump horsepower in a cell of its own county must be >= ``GATE_HP_AGREEMENT``,
        and every cell whose horsepower majority disagrees with its flag must lie mostly
        (by area) in its flagged county with each disagreeing pump within
        ``GATE_BORDER_M`` of the county line, those cells holding at most
        ``GATE_DISAGREE_HP_MAX`` of the horsepower;
    (b) every station of ``stations`` labelled 彰化縣 lies in a Changhua cell and every one
        labelled 雲林縣 in a rest cell.

    A source that is missing is reported as not run (and fails the gate)."""
    import pandas as pd
    import shapely

    flags = np.asarray(flags).astype(bool)
    cty = _counties(gpkg)
    ch_geom = cty[cty["county"] == _norm(CHANGHUA)].union_all()
    out: dict = {}

    def cells_of(x, y) -> np.ndarray:
        return np.array([-1 if (i := grid.active_index(float(a), float(b))) is None else i
                         for a, b in zip(x, y, strict=True)], dtype="int64")

    if os.path.exists(census):
        pm = pd.read_parquet(census, columns=["PUMP_HP", "TWD97_X", "TWD97_Y", "county"])
        pm = pm[pm["county"].isin(["changhua", "yunlin"])].dropna(subset=["TWD97_X",
                                                                          "TWD97_Y"])
        cell = cells_of(pm["TWD97_X"].to_numpy(), pm["TWD97_Y"].to_numpy())
        pm = pm.assign(cell=cell)[cell >= 0]
        hp = pm["PUMP_HP"].fillna(0.0).to_numpy(dtype="float64")
        lab = (pm["county"] == "changhua").to_numpy()
        fl = flags[pm["cell"].to_numpy()]
        agree = float(hp[lab == fl].sum() / max(hp.sum(), 1e-12))
        per = pd.DataFrame({"cell": pm["cell"].to_numpy(), "hp": hp, "ch_hp": hp * lab})
        per = per.groupby("cell")[["hp", "ch_hp"]].sum()
        maj = (per["ch_hp"] > 0.5 * per["hp"]).to_numpy()
        cells = per.index.to_numpy()
        bad = cells[maj != flags[cells]]
        c = grid.centroids()
        half = float(grid.dx) / 2.0
        bad_rec = []
        for b in bad.tolist():
            box = shapely.box(c[b, 0] - half, c[b, 1] - half, c[b, 0] + half, c[b, 1] + half)
            land = shapely.area(shapely.intersection(box, cty.union_all()))
            frac_ch = shapely.area(shapely.intersection(box, ch_geom)) / max(land, 1e-12)
            sel = (pm["cell"].to_numpy() == b) & (lab != flags[b])
            dist = shapely.distance(ch_geom.boundary, shapely.points(
                pm["TWD97_X"].to_numpy()[sel], pm["TWD97_Y"].to_numpy()[sel]))
            bad_rec.append({"cell": int(b), "flag_changhua": bool(flags[b]),
                            "hp": float(per.loc[b, "hp"]),
                            "changhua_hp_share": float(per.loc[b, "ch_hp"] / per.loc[b, "hp"])
                            if per.loc[b, "hp"] > 0 else None,
                            "flagged_county_area_share": float(frac_ch if flags[b]
                                                               else 1.0 - frac_ch),
                            "max_disagreeing_pump_to_line_m": float(np.max(dist))
                            if dist.size else 0.0})
        border_ok = all(r["flagged_county_area_share"] > 0.5
                        and r["max_disagreeing_pump_to_line_m"] <= GATE_BORDER_M
                        for r in bad_rec)
        bad_hp = float(sum(r["hp"] for r in bad_rec) / max(hp.sum(), 1e-12))
        border_ok = border_ok and bad_hp <= GATE_DISAGREE_HP_MAX
        out["pump_census"] = {"path": census, "n_pumps_in_grid": int(len(pm)),
                              "hp_agreement": agree, "hp_agreement_min": GATE_HP_AGREEMENT,
                              "cells_with_pumps": int(len(cells)),
                              "cells_hp_majority_disagrees": bad_rec,
                              "border_m": GATE_BORDER_M,
                              "disagreeing_cells_hp_share": bad_hp,
                              "disagreeing_cells_hp_share_max": GATE_DISAGREE_HP_MAX,
                              "passed": bool(agree >= GATE_HP_AGREEMENT and border_ok)}
    else:
        out["pump_census"] = {"path": census, "passed": False, "reason": "not found"}
    if os.path.exists(stations):
        st = pd.read_parquet(stations).drop_duplicates("sid")
        xy = st["LocationByTWD97"].astype(str).str.extract(r"([\d.]+)\D+([\d.]+)")
        xy = xy.astype(float)
        st = st.assign(x=xy[0].to_numpy(), y=xy[1].to_numpy()).dropna(subset=["x", "y"])
        cell = cells_of(st["x"].to_numpy(), st["y"].to_numpy())
        st = st.assign(cell=cell)[cell >= 0]
        county = st["CountyName"].map(_norm).to_numpy()
        f = flags[st["cell"].to_numpy()]
        ch_bad = st["sid"].to_numpy()[(county == _norm(CHANGHUA)) & ~f].tolist()
        yl_bad = st["sid"].to_numpy()[(county == _norm("雲林縣")) & f].tolist()
        out["stations"] = {"path": stations, "n_in_grid": int(len(st)),
                           "changhua_labelled": int((county == _norm(CHANGHUA)).sum()),
                           "yunlin_labelled": int((county == _norm("雲林縣")).sum()),
                           "changhua_label_in_rest_cell": ch_bad,
                           "yunlin_label_in_changhua_cell": yl_bad,
                           "passed": not ch_bad and not yl_bad}
    else:
        out["stations"] = {"path": stations, "passed": False, "reason": "not found"}
    out["passed"] = bool(out["pump_census"]["passed"] and out["stations"]["passed"])
    return out


def build_county_map(grid, gpkg: str = COUNTY_GPKG) -> np.ndarray:
    """The Changhua flag of every active cell of ``grid`` (the module's rule)."""
    return changhua_flags(grid.centroids(), gpkg)


def save_county_npz(path: str, grid, flags: np.ndarray, polygon: str,
                    gpkg: str = COUNTY_GPKG) -> None:
    flags = np.asarray(flags).astype(bool)
    if flags.shape != (grid.n_active,):
        raise ValueError(f"flags {flags.shape} do not match {grid.n_active} active cells")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, changhua=flags.astype("uint8"), centroids=grid.centroids(),
             n_active=np.int64(grid.n_active), sha256=np.str_(county_sha256(flags)),
             county=np.str_(CHANGHUA), county_field=np.str_(COUNTY_FIELD),
             gpkg=np.str_(gpkg), polygon=np.str_(polygon), rule=np.str_(RULE),
             **_grid_record(grid))


def load_county_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    flags = np.asarray(out["changhua"]).astype(bool)
    if county_sha256(flags) != str(out["sha256"]):
        raise ValueError(f"{path}: stored sha256 does not match its own flags (corrupt)")
    out["changhua"] = flags
    return out


def county_flags_for_grid(grid, path: str | None = DEFAULT_MAP, gpkg: str = COUNTY_GPKG,
                          expect_sha256: str | None = None) -> tuple[np.ndarray, str]:
    """``(flags, source)``: the cached map when ``path`` holds one built on this very grid,
    else rebuilt from the gpkg. ``expect_sha256`` (a calibration's ``zone_county_sha256``)
    must match, or this raises: a consumer never runs on a county map other than the one
    its calibration used."""
    flags = None
    source = "rebuilt from " + gpkg
    if path and os.path.exists(path):
        z = load_county_npz(path)
        if _same_grid(z, grid):
            flags, source = z["changhua"], path
    if flags is None:
        flags = build_county_map(grid, gpkg)
    if expect_sha256 is not None and county_sha256(flags) != str(expect_sha256):
        raise ValueError(f"the county map ({source}) has sha256 {county_sha256(flags)[:12]}, "
                         f"the calibration recorded {str(expect_sha256)[:12]}: a different "
                         "county map (or grid) than the one the parameters were fitted on")
    return flags, source


def verify_county_cache(grid, path: str = DEFAULT_MAP, gpkg: str = COUNTY_GPKG
                        ) -> np.ndarray:
    """``calibrate_flow --zone-county-split``: the cached map must exist, be built on this
    grid, and equal a fresh rebuild from the gpkg; returns its flags."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"--zone-county-map {path!r} does not exist; build it with "
                                "python -m hydrophysics.twin.zone_county build")
    z = load_county_npz(path)
    if not _same_grid(z, grid):
        raise ValueError(f"{path} was built on another grid (dx {float(z['dx']):g}, "
                         f"{int(z['n_active'])} cells) than this run's ({grid.dx:g}, "
                         f"{grid.n_active})")
    fresh = build_county_map(grid, gpkg)
    if not np.array_equal(fresh, z["changhua"]):
        raise ValueError(f"{path} differs from a rebuild from {gpkg} in "
                         f"{int((fresh != z['changhua']).sum())} cells: stale cache")
    return z["changhua"]


def county_meta(path: str, flags: np.ndarray) -> dict:
    """The theta-meta record of a county-split calibration (``META_KEYS``)."""
    return {"zone_county_split": True, "zone_county_map": path,
            "zone_county_sha256": county_sha256(flags),
            "zone_county_npz_sha256": _file_sha256(path) if os.path.exists(path) else "",
            "zone_county_n_changhua": int(np.asarray(flags).astype(bool).sum())}


def county_flags_from_meta(grid, meta: dict | None) -> np.ndarray | None:
    """``None`` unless the calibration recorded ``zone_county_split``; then its county map
    for ``grid``, verified against the recorded ``zone_county_sha256``."""
    meta = meta or {}
    if not meta.get("zone_county_split"):
        return None
    sha = meta.get("zone_county_sha256")
    if not sha:
        raise ValueError("meta says zone_county_split but records no zone_county_sha256; "
                         "the county map cannot be verified")
    flags, _ = county_flags_for_grid(grid, meta.get("zone_county_map") or DEFAULT_MAP,
                                     expect_sha256=sha)
    return flags


def tied_copy_keys(theta: dict, meta: dict | None) -> dict[str, str]:
    """``{copy key: parent key}`` of the Changhua copies a ``--county-tie`` calibration
    tied to their parents (``zone_county_tied`` in the meta: base zone names); empty
    otherwise."""
    tied = (meta or {}).get("zone_county_tied") or []
    if not tied:
        return {}
    from .calibrate_flow import _key_zone

    zones = {f"{z}{COUNTY_SUFFIX}" for z in tied}
    return {k: k[: -len(COUNTY_SUFFIX)] for k in theta
            if _key_zone(k) in zones and k[: -len(COUNTY_SUFFIX)] in theta}


def apply_county_ties(theta: dict, meta: dict | None) -> dict:
    """``theta`` with every tied copy (``tied_copy_keys``) set to its parent's value: a
    perturbed parent (``uncertainty``'s Jacobian, a posterior draw) moves its tied copy
    with it, as in the calibration. A no-op (the same dict) without ties."""
    pairs = tied_copy_keys(theta, meta)
    if not pairs:
        return theta
    return {k: (theta[pairs[k]] if k in pairs else v) for k, v in theta.items()}


def free_copy_pairs(theta: dict, meta: dict | None) -> list[tuple[str, str]]:
    """``[(copy key, parent key)]`` of the free (untied) Changhua copies of a county-split
    theta, for ``--county-copy-sd``'s prior."""
    if not (meta or {}).get("zone_county_split"):
        return []
    from .calibrate_flow import _key_zone

    tied = tied_copy_keys(theta, meta)
    return [(k, k[: -len(COUNTY_SUFFIX)]) for k in theta
            if (_key_zone(k) or "").endswith(COUNTY_SUFFIX) and k not in tied
            and k[: -len(COUNTY_SUFFIX)] in theta]


# ---------------------------------------------------------------------------------------
# zone maps from a calibration's meta: the one place consumers rebuild them
# ---------------------------------------------------------------------------------------
def _boundaries(meta: dict) -> tuple:
    from .calibrate_flow import _parse_zone_boundaries

    return _parse_zone_boundaries(meta.get("zone_boundaries", "205,182"), allow_split=True)


def flow_zone_map(grid, meta: dict | None) -> tuple[np.ndarray, tuple[str, ...]]:
    """The flow model's zone ids per active cell and their names (``zone_labels``): the
    easting zones of ``zone_boundaries`` (with the proximal split when recorded), each
    split by county when the calibration used ``--zone-county-split``."""
    meta = meta or {}
    p, d, s = _boundaries(meta)
    z = fan_zones(grid.centroids(), proximal_km=p, distal_km=d, split_km=s)
    ch = county_flags_from_meta(grid, meta)
    if ch is not None:
        z = county_split(z, ch, N_ZONES + int(s is not None))
    return z, zone_labels(s is not None, ch is not None)


def column_zone_map(grid, meta: dict | None) -> np.ndarray:
    """The compaction column's (and scenario's) zone ids per active cell: three easting
    zones (the proximal split collapsed), and with ``--zone-county-split`` the Changhua
    copy of each (ids 3-5), so a zonal column fitted on a county-split flow model gives
    Changhua its own column parameters."""
    meta = meta or {}
    p, d, _ = _boundaries(meta)
    z = fan_zones(grid.centroids(), proximal_km=p, distal_km=d)
    ch = county_flags_from_meta(grid, meta)
    return z if ch is None else county_split(z, ch, N_ZONES)


def column_zone_weights(grid, meta: dict | None):
    """``km -> (n_zones, A)`` blend weights of the column's zones (``zone_blend_weights``
    on the meta's lines, split by county when the calibration was)."""
    meta = meta or {}
    p, d, _ = _boundaries(meta)
    cent = grid.centroids()
    ch = county_flags_from_meta(grid, meta)

    def weights(km: float) -> np.ndarray:
        w = zone_blend_weights(cent, p, d, km)
        return w if ch is None else county_split_weights(w, ch)

    return weights


def flow_zone_weights(grid, meta: dict | None, blend_km: float) -> np.ndarray:
    """The flow model's ``--zone-blend-km`` weights ``(n_zones, A)``, split by county when
    the calibration was."""
    meta = meta or {}
    p, d, s = _boundaries(meta)
    w = zone_blend_weights(grid.centroids(), p, d, float(blend_km), split_km=s)
    ch = county_flags_from_meta(grid, meta)
    return w if ch is None else county_split_weights(w, ch)


def column_zone_names(meta: dict | None) -> tuple[str, ...]:
    """The column's zone names for a calibration meta (3, or 6 with the county split)."""
    return zone_labels(False, bool((meta or {}).get("zone_county_split")))


# ---------------------------------------------------------------------------------------
# the product
# ---------------------------------------------------------------------------------------
def _staging_manifest():
    """The dataprep staging library (``dataprep_staging/common``), found from the
    repository root (the working directory, or ``DATAPREP_REPO``)."""
    root = os.environ.get("DATAPREP_REPO") or os.getcwd()
    staging = os.path.join(root, "dataprep_staging")
    if not os.path.isdir(os.path.join(staging, "common")):
        raise SystemExit(f"no dataprep staging library under {staging}; run from the "
                         "repository root")
    if staging not in sys.path:
        sys.path.insert(0, staging)
    from common import manifest  # noqa: PLC0415

    return manifest


def _code_state() -> dict:
    """Where the building code came from: the git branch/HEAD of its checkout and whether
    the two code files are committed (the manifest also records their sha256)."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a) -> str:
        try:
            return subprocess.run(["git", "-C", here, *a], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    status = git("status", "--porcelain", "--", "zone_county.py", "zones.py")
    return {"checkout": git("rev-parse", "--show-toplevel"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "uncommitted": status.splitlines()}


def build_product(dx: float = 1000.0, polygon: str | None = None,
                  gpkg: str = COUNTY_GPKG, log=print, census: str = PUMP_CENSUS,
                  stations: str = STATIONS) -> str:
    """Build ``results/dataprep/zone_county_map/v<N>/`` (npz, validation.json, MANIFEST),
    gated on the independent county labels (``independent_label_gates``) and on the
    off-polygon cells lying within ``GATE_NO_POLYGON_MAX_M`` of land."""
    from .calibrate_flow import DEFAULT_PATHS
    from .grid import build_grid

    polygon = polygon or DEFAULT_PATHS["polygon"]
    grid = build_grid(polygon, dx=dx)
    flags = build_county_map(grid, gpkg)
    val = county_diagnostics(grid, flags, gpkg)
    val["sha256_flags"] = county_sha256(flags)
    val["rule"] = RULE
    empty = [k for k, v in val["per_zone"].items()
             if v["changhua"] == 0 or v["cells"] == v["changhua"]]
    manifest = _staging_manifest()
    d = manifest.new_version_dir(PRODUCT)
    npz = os.path.join(str(d), NPZ_NAME)
    save_county_npz(npz, grid, flags, polygon, gpkg)
    back = load_county_npz(npz)
    rt_ok = bool(np.array_equal(back["changhua"], flags) and _same_grid(back, grid))
    rebuild_ok = bool(np.array_equal(build_county_map(grid, gpkg), flags))
    gates = independent_label_gates(grid, flags, gpkg, census, stations)
    near_ok = val["no_county_cells_max_distance_m"] <= GATE_NO_POLYGON_MAX_M
    val.update({"round_trip_ok": rt_ok, "rebuild_reproduces": rebuild_ok,
                "zones_without_changhua_or_rest": empty, "independent_labels": gates,
                "no_county_cells_within_max_m": bool(near_ok),
                "no_county_max_m": GATE_NO_POLYGON_MAX_M, "code_state": _code_state()})
    val["passed"] = bool(rt_ok and rebuild_ok and not empty and flags.any()
                         and (~flags).any() and gates["passed"] and near_ok)
    with open(os.path.join(str(d), "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(val, fh, ensure_ascii=False, indent=1)
    log(json.dumps(val, ensure_ascii=False, indent=1))
    if not val["passed"]:
        raise SystemExit(f"validation failed; {d} left without a manifest")
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L1",
        inputs=[gpkg, polygon, census, stations],
        code=[os.path.abspath(__file__), os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "zones.py")],
        parameters={"rule": RULE, "dx_m": float(dx), "county_field": COUNTY_FIELD,
                    "county": CHANGHUA, "crs": (f"gpkg coordinates read as EPSG:{GPKG_TRUE_EPSG} (its EPSG:3828 "
                            "tag overridden: GADM projected to TM2, mis-tagged; v1 "
                            "reprojected it and was shifted ~850 m)"),
                    "validation_labels": "pump census county (address) and WRA station "
                                         "CountyName: gates only, never fitted",
                    "hash": "sha256 of the per-cell uint8 flag in active-cell order "
                            "(zone_county_sha256 in a calibration's meta)",
                    "consumer": "calibrate_flow --zone-county-split (rung R1c)"},
        fit_window_end="2022-12",
        validation_report="validation.json",
        gate={"passed": True, "criteria": (
            "every active cell labelled; npz round trip and a rebuild from the gpkg reproduce "
            "the flags; every zone of 205,182,208 has Changhua and rest cells; pump census "
            f"horsepower in its own county's cells >= {GATE_HP_AGREEMENT:.0%} and every "
            "cell whose horsepower majority disagrees lies mostly in its flagged county with "
            f"the disagreeing pumps within {GATE_BORDER_M:g} m of the line, together <= "
            f"{GATE_DISAGREE_HP_MAX:.1%} of the horsepower; every WRA "
            "station labelled 彰化縣 in a Changhua cell and every 雲林縣 one in a rest cell; "
            f"off-polygon cells within {GATE_NO_POLYGON_MAX_M:g} m of land")},
        allowed_uses=["model_design", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target"],
        personal_data="none (administrative boundaries only)",
        extra={"time_invariance": "administrative boundary only, no time-dependent data; "
                                  "the Changhua County boundary was not changed by the "
                                  "2010 or 2014 municipal mergers",
               "supersedes": ("v1 (built 2026-10-15 with a spurious TWD67->TWD97 shift of "
                              "the polygons and off-polygon coastal cells sent to rest; "
                              "42 cells mislabelled)"),
               "code_state": val["code_state"]})
    manifest.write_manifest(d, m)
    log(f"written {d}")
    return npz


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="per-cell county map for --zone-county-split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build results/dataprep/zone_county_map/v<N>")
    b.add_argument("--dx", type=float, default=1000.0)
    b.add_argument("--polygon", default=None)
    b.add_argument("--gpkg", default=COUNTY_GPKG)
    b.add_argument("--census", default=PUMP_CENSUS,
                   help="pump census with a county column (validation gate only)")
    b.add_argument("--stations", default=STATIONS,
                   help="WRA station table with CountyName (validation gate only)")
    v = sub.add_parser("verify", help="rebuild from the gpkg and compare with a cached map")
    v.add_argument("--map", default=DEFAULT_MAP)
    v.add_argument("--dx", type=float, default=1000.0)
    v.add_argument("--polygon", default=None)
    v.add_argument("--gpkg", default=COUNTY_GPKG)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build_product(args.dx, args.polygon, args.gpkg, census=args.census,
                      stations=args.stations)
        return
    from .calibrate_flow import DEFAULT_PATHS
    from .grid import build_grid

    grid = build_grid(args.polygon or DEFAULT_PATHS["polygon"], dx=args.dx)
    flags = verify_county_cache(grid, args.map, args.gpkg)
    print(f"{args.map}: {int(flags.sum())} Changhua cells of {grid.n_active}; rebuild "
          f"identical; sha256 {county_sha256(flags)}")


if __name__ == "__main__":
    main()

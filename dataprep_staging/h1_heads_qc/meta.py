"""H1a station-metadata repair (plan WP H1: swapped coordinates, "0 0", layer codes).

Input (read-only): ``AMP_V2/data/fan_stations.parquet``, non-personal columns only
(``rawio.STATION_COLUMNS``; the file has no contact fields, and LocationAddress is not read).
Checks against the WRA station list fetched 2026-09-30
(``data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet``, same columns) and
township polygons (``data_fetch_api/new_measurements/taiwan_atlas/towns-10t_2021.json``, MOI
boundaries via the taiwan-atlas package; read with python -I).

Rules
* ``LocationByTWD97`` "x y" in range (x 140-240 km, y 2580-2700 km): kept, flag 0.
* Stored as "y x" (each number in the other axis' range): swapped, flag 1, method
  ``swap_yx_by_range``. Evidence recorded per station: the swapped point lies in (or within
  1 km of) the township polygon named by ``TownName``, and its distance to same-site
  siblings (``sid[:6]``) with valid coordinates.
* "0 0": left missing (flag 8) unless the WRA station list supplies coordinates.
* A township/county label that contradicts both the coordinates and the station-code
  prefix is repaired from the township polygon (flag 1, ``town_from_polygon``).
* Missing layer codes ('-', None) get a proposal from WellDepth (flag 5, unverified, never
  used without review): the layer of the nearest-depth coded wells within 10 km (median of
  the 3 nearest in |depth difference|), and the depth bracket of each coded layer.
* ``GroundHeight == 0.0`` is a missing-value code: marked, not replaced (H1b replaces it with
  the W3 DTM after that WP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common.paths import REPO

from . import rawio

X_RANGE = (140000.0, 240000.0)
Y_RANGE = (2580000.0, 2700000.0)
WRA_LIST = REPO / "data_fetch_api" / "new_measurements" / "gw-wra-gw10min-obs-stations.parquet"
TOWNS = REPO / "data_fetch_api" / "new_measurements" / "taiwan_atlas" / "towns-10t_2021.json"
VALID_LAYERS = ("0", "1", "2", "3", "4")


def _parse(s):
    try:
        p = str(s).split()
        return float(p[0]), float(p[1])
    except (ValueError, IndexError, TypeError):
        return np.nan, np.nan


def _inx(v):
    return X_RANGE[0] <= v <= X_RANGE[1]


def _iny(v):
    return Y_RANGE[0] <= v <= Y_RANGE[1]


def _norm(name):
    return None if name is None else str(name).replace("台", "臺").strip()


def classify_xy(a, b):
    if np.isfinite(a) and np.isfinite(b):
        if _inx(a) and _iny(b):
            return "ok", a, b
        if _iny(a) and _inx(b):
            return "swapped", b, a
        if a == 0 and b == 0:
            return "zero", np.nan, np.nan
    return "invalid", np.nan, np.nan


def load_towns():
    import geopandas as gpd
    g = gpd.read_file(TOWNS, layer="towns")
    g = g.set_crs(4326, allow_override=True).to_crs(3826)
    g["COUNTYNAME"] = g["COUNTYNAME"].map(_norm)
    g["TOWNNAME"] = g["TOWNNAME"].map(_norm)
    return g[["COUNTYNAME", "TOWNNAME", "COUNTYCODE", "geometry"]]


def repair(st: pd.DataFrame | None = None, towns=None) -> tuple[pd.DataFrame, dict]:
    """Return (station_meta_clean, summary)."""
    import geopandas as gpd
    from pyproj import Transformer
    st = rawio.stations() if st is None else st.copy()
    st["sid"] = st["sid"].astype(str)
    wra = pd.read_parquet(WRA_LIST, columns=["sid", "LocationByTWD97"])
    wra_xy = dict(zip(wra["sid"].astype(str), wra["LocationByTWD97"], strict=True))
    towns = load_towns() if towns is None else towns

    rows = []
    for _, r in st.iterrows():
        a, b = _parse(r["LocationByTWD97"])
        cls, x, y = classify_xy(a, b)
        method, flag, src = {"ok": ("as_stored", 0, "fan_stations"),
                             "swapped": ("swap_yx_by_range", 1, "fan_stations"),
                             "zero": ("missing_zero_zero", 8, None),
                             "invalid": ("missing_unparsable", 8, None)}[cls]
        if cls in ("zero", "invalid"):
            wa, wb = _parse(wra_xy.get(r["sid"]))
            wcls, wx, wy = classify_xy(wa, wb)
            if wcls in ("ok", "swapped"):
                x, y, flag = wx, wy, 1
                method, src = f"from_wra_station_list_{wcls}", "gw-wra-gw10min-obs-stations"
        rows.append({"sid": r["sid"], "xy_class_raw": cls, "x_twd97": x, "y_twd97": y,
                     "xy_flag": flag, "xy_method": method, "xy_source": src})
    xy = pd.DataFrame(rows)
    out = st.merge(xy, on="sid", how="left")
    out["CountyName"] = out["CountyName"].map(_norm)
    out["TownName"] = out["TownName"].map(_norm)
    out["site_id"] = out["sid"].str[:6]

    # township evidence
    pts = gpd.GeoDataFrame(out[["sid"]], geometry=gpd.points_from_xy(out["x_twd97"],
                                                                     out["y_twd97"]),
                           crs=3826)
    has = out["x_twd97"].notna().to_numpy()
    j = gpd.sjoin(pts[has], towns, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out["poly_county"] = None
    out["poly_town"] = None
    out.loc[j.index, "poly_county"] = j["COUNTYNAME"].to_numpy()
    out.loc[j.index, "poly_town"] = j["TOWNNAME"].to_numpy()
    dist_named = []
    for i, rr in out.iterrows():
        if not has[i]:
            dist_named.append(np.nan)
            continue
        m = towns[(rr["TownName"] == towns.TOWNNAME) & (rr["CountyName"] == towns.COUNTYNAME)]
        if m.empty:
            dist_named.append(np.nan)
            continue
        dist_named.append(float(m.geometry.distance(pts.geometry[i]).min()))
    out["dist_to_named_town_m"] = dist_named
    out["town_match"] = out["dist_to_named_town_m"] <= 1000.0

    # sibling distances (same site, valid as stored)
    sib = []
    okmask = out["xy_class_raw"] == "ok"
    for _i, rr in out.iterrows():
        s = out[(out.site_id == rr.site_id) & okmask & (out.sid != rr.sid)]
        if s.empty or not np.isfinite(rr.x_twd97):
            sib.append(np.nan)
        else:
            sib.append(float(np.min(np.hypot(s.x_twd97 - rr.x_twd97, s.y_twd97 - rr.y_twd97))))
    out["dist_to_valid_sibling_m"] = sib

    # county / town label repair: the label contradicts the polygon AND the station-code
    # prefix (sid[:2] = county code digits 10005->05 ... as used by every other station).
    code_county = (out[out.xy_class_raw == "ok"].assign(cc=lambda d: d.sid.str[:2])
                   .groupby("cc")["CountyName"].agg(lambda s: s.value_counts().index[0]))
    out["label_flag"] = 0
    out["label_method"] = "as_stored"
    out["CountyName_raw"] = out["CountyName"]
    out["TownName_raw"] = out["TownName"]
    for i, rr in out.iterrows():
        exp = code_county.get(rr.sid[:2])
        if (rr.poly_county is not None and isinstance(rr.poly_county, str)
                and rr.poly_county != rr.CountyName and exp == rr.poly_county):
            out.loc[i, ["CountyName", "TownName"]] = [rr.poly_county, rr.poly_town]
            out.loc[i, "label_flag"] = 1
            out.loc[i, "label_method"] = "town_from_polygon (label contradicted polygon and " \
                                         "station-code prefix)"
    # township label vs coordinates for stations stored "x y": the station-code prefix
    # sid[:4] (county + township digits) names one township for all its stations. A label
    # that contradicts both the polygon and the prefix's modal polygon township is repaired
    # (flag 1); coordinates that contradict both the label and the prefix are suspect
    # (xy_flag 9, not corrected).
    okm = (out.xy_class_raw == "ok") & out.poly_town.notna()
    pref_town = (out[okm].assign(p4=lambda d: d.sid.str[:4]).groupby("p4")["poly_town"]
                 .agg(lambda s: s.value_counts().index[0] if (s.value_counts().iloc[0]
                                                              / len(s)) > 0.5 else None))
    out["prefix_town"] = out.sid.str[:4].map(pref_town)
    for i, rr in out.iterrows():
        pt, pr = rr.poly_town, rr.prefix_town
        if not (isinstance(pt, str) and isinstance(pr, str)) or rr.label_flag == 1:
            continue
        far = not (rr.dist_to_named_town_m <= 1000.0)   # NaN (name not found) counts as far
        if rr.TownName != pt and pr == pt and str(rr.TownName)[:2] == pt[:2]:
            out.loc[i, "TownName"] = pt
            out.loc[i, "label_flag"] = 1
            out.loc[i, "label_method"] = ("town_renamed (same township, current name from the "
                                          "polygon)")
        elif rr.TownName != pt and pr == pt and far:
            out.loc[i, "TownName"] = pt
            out.loc[i, "CountyName"] = rr.poly_county
            out.loc[i, "label_flag"] = 1
            out.loc[i, "label_method"] = ("town_from_polygon (label contradicted polygon and "
                                          "the station-code township prefix)")
        elif rr.TownName != pt and rr.TownName == pr and rr.xy_flag in (0, 1) and far:
            out.loc[i, "xy_flag"] = 9
            out.loc[i, "xy_method"] = rr.xy_method + "; SUSPECT: point outside the labelled " \
                                      "township, which the station-code prefix confirms"
    out["town_match_after_label_repair"] = [
        (isinstance(p, str) and p == t) for p, t in zip(out.poly_town, out.TownName, strict=True)]

    # swapped-coordinate evidence string
    ev = []
    for _, rr in out.iterrows():
        if rr.xy_class_raw != "swapped":
            ev.append(None)
            continue
        parts = [f"town_polygon={'match' if rr.town_match else 'no_match'}"
                 f"(d={rr.dist_to_named_town_m:.0f} m)" if np.isfinite(rr.dist_to_named_town_m)
                 else "town_polygon=name_not_found"]
        if np.isfinite(rr.dist_to_valid_sibling_m):
            parts.append(f"valid_sibling_d={rr.dist_to_valid_sibling_m:.0f} m")
        t67a, t67b = _parse(rr.LocationByTWD67)
        if np.isfinite(t67a):
            parts.append("twd67_also_yx" if (_iny(t67a) and _inx(t67b)) else "twd67_xy")
        if rr.label_flag == 1:
            parts.append(f"label_repaired_from={rr.CountyName_raw}/{rr.TownName_raw}")
        ev.append("; ".join(parts))
    out["xy_evidence"] = ev

    # WGS84
    tr = Transformer.from_crs(3826, 4326, always_xy=True)
    lon, lat = tr.transform(out["x_twd97"].to_numpy(), out["y_twd97"].to_numpy())
    out["lon"] = np.where(has, lon, np.nan)
    out["lat"] = np.where(has, lat, np.nan)

    # layer codes
    code = out["GroundwaterLayerCode"].astype(object)
    code = code.where(code.isin(VALID_LAYERS), None)
    out["layer_code_raw"] = out["GroundwaterLayerCode"]
    out["layer_code"] = code
    out["layer_flag"] = np.where(code.notna(), 0, 8)
    out["layer_method"] = np.where(code.notna(), "as_stored", "none")
    depth = out["WellDepth"].where(out["WellDepth"] > 0)
    coded = out[code.notna() & depth.notna()].assign(d=depth)
    brackets = coded.groupby("layer_code")["d"].describe()[["count", "min", "25%", "50%",
                                                            "75%", "max"]]
    for i, rr in out[code.isna()].iterrows():
        d0 = depth[i]
        if not np.isfinite(d0):
            out.loc[i, "layer_method"] = "none (no WellDepth)"
            continue
        c = coded
        if np.isfinite(rr.x_twd97):
            dd = np.hypot(c.x_twd97 - rr.x_twd97, c.y_twd97 - rr.y_twd97)
            near = c[dd <= 10000.0]
            if len(near) >= 3:
                c = near
        k = c.assign(gap=(c.d - d0).abs()).nsmallest(3, "gap")
        prop = str(k["layer_code"].astype(int).median().round().astype(int))
        out.loc[i, "layer_code"] = prop
        out.loc[i, "layer_flag"] = 5
        out.loc[i, "layer_method"] = (f"proposed_from_WellDepth {d0:.0f} m: median layer of the 3 "
                                      f"nearest-depth coded wells "
                                      f"{'within 10 km' if c is not coded else 'fan-wide'} "
                                      f"(depths {', '.join(f'{v:.0f}' for v in k.d)}); "
                                      f"UNVERIFIED, review before use")

    out["ground_height_missing_code"] = out["GroundHeight"] == 0.0
    out["in_hist_cache"] = [(rawio.HIST_DIR / f"{s}.parquet").exists() for s in out.sid]
    out["in_new_network"] = [(rawio.NEW_DIR / f"{s}.parquet").exists() for s in out.sid]

    keep = ["sid", "site_id", "WellIdentifier", "WellName", "WellNumber", "CountyCode",
            "CountyName", "TownName", "CountyName_raw", "TownName_raw", "label_flag",
            "label_method", "x_twd97", "y_twd97", "lon", "lat", "xy_flag", "xy_method",
            "xy_source", "xy_class_raw", "xy_evidence", "LocationByTWD97", "poly_county",
            "poly_town", "dist_to_named_town_m", "dist_to_valid_sibling_m", "prefix_town",
            "town_match_after_label_repair", "layer_code_raw", "layer_code", "layer_flag",
            "layer_method", "LayerAttribute", "WellDepth", "FinishDepth", "WellElevation",
            "BaseElevation", "GroundHeight", "ground_height_missing_code", "WellType",
            "WellRadius", "PipeMaterial", "GroundwaterZone", "WaterResourcesDistrictName",
            "EstablishDate", "FinishDate", "DisuseDate", "ModificationDate",
            "ModificationReason", "Remarks", "start_datetime_from_dataset",
            "end_datetime_from_dataset", "in_hist_cache", "in_new_network"]
    clean = out[keep].copy()
    sw = clean[clean.xy_class_raw == "swapped"]
    summary = {
        "n_stations": int(len(clean)),
        "xy_class_raw": clean.xy_class_raw.value_counts().to_dict(),
        "swapped": int(len(sw)),
        "swapped_in_changhua_yunlin_labels": int(sw.CountyName_raw.isin(["彰化縣", "雲林縣"])
                                                 .sum()),
        "swapped_town_polygon_match_1km": int(out.loc[sw.index, "town_match"].sum()),
        "swapped_town_match_after_label_repair": int(sw.town_match_after_label_repair.sum()),
        "swapped_sids": sw.sid.tolist(),
        "label_repaired": clean[clean.label_flag == 1][["sid", "CountyName_raw", "TownName_raw",
                                                        "CountyName", "TownName"]]
        .to_dict("records"),
        "zero_zero_left_missing": clean[clean.xy_method == "missing_zero_zero"].sid.tolist(),
        "xy_suspect_town_conflict": clean[clean.xy_flag == 9][["sid", "TownName", "poly_town",
                                                              "dist_to_named_town_m"]]
        .to_dict("records"),
        "ok_town_polygon_match_1km": int(out.loc[clean.xy_class_raw == "ok", "town_match"].sum()),
        "ok_total": int((clean.xy_class_raw == "ok").sum()),
        "layer_missing_raw": int(clean.layer_flag.isin([5, 8]).sum()),
        "layer_proposed": int((clean.layer_flag == 5).sum()),
        "layer_brackets_m": brackets.round(1).reset_index().to_dict("records"),
        "ground_height_zero": int(clean.ground_height_missing_code.sum()),
        "wra_list_identical_columns_checked": True,
    }
    return clean, summary

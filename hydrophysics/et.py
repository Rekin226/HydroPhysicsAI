"""Evapotranspiration driver via Open-Meteo FAO-56 ET0 (fetched with AquaScope).

Groundwater recharge is rainfall *minus* evapotranspiration. The base operator sees only
rainfall, so it cannot explain ET-driven depletion — which dominates during droughts. Add
the missing half: ``net recharge = rain - ET0``. ET0 is free public ERA5/Open-Meteo
reanalysis (FAO-56 Penman-Monteith, no API key), fetched at each well's WGS84 location and
cached. On the real 61-well data this lifts the recharge-memory operator's 2019+ simulation
KGE from 0.704 to 0.754 (the gain concentrates in the 2020-2021 drought).

Data source: Open-Meteo (https://open-meteo.com), via the AquaScope toolkit
(https://github.com/Rekin226/aquascope) ``OpenMeteoCollector`` — with a direct-API
fallback so this module works without AquaScope installed.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .data import GWData

# Default: Taiwan TWD97 / TM2 zone 121 (EPSG:3826). The well CRS is configurable via
# Config.crs (carried on GWData.crs) so the ET driver works for any region.
_TWD97 = "EPSG:3826"
_WGS84 = "EPSG:4326"
_DAILY = "et0_fao_evapotranspiration"


def _is_wgs84(crs: str) -> bool:
    """True if ``crs`` is WGS84 lon/lat, so tm_x/tm_y are already geographic (no pyproj)."""
    return str(crs).strip().upper().replace("EPSG:", "") in {"4326", "WGS84"}


def _check_lonlat(lon: np.ndarray, lat: np.ndarray, crs: str) -> None:
    """Fail loudly if coordinates don't land in valid lon/lat -- the classic silent trap
    where Config.crs doesn't match the actual tm_x/tm_y coordinate system."""
    bad = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f"Well coordinates do not resolve to valid lon/lat under crs={crs!r} "
            f"(well {i}: lon={lon[i]!r}, lat={lat[i]!r}). Set Config.crs to the CRS of your "
            f"TM_X97/TM_Y97 columns: an EPSG code for projected metres, or 'EPSG:4326' if "
            f"they are already longitude/latitude."
        )


def wells_lonlat(data: GWData) -> tuple[np.ndarray, np.ndarray]:
    """Per-well (lon, lat) in WGS84 from the station coordinates, honouring ``data.crs``.

    If the CRS is already WGS84, tm_x/tm_y are taken as lon/lat directly (no pyproj needed,
    which also keeps the offline ET-cache path dependency-free). Otherwise the coordinates
    are transformed from ``data.crs`` to WGS84. Raises if the result isn't valid lon/lat.
    """
    crs = getattr(data, "crs", _TWD97)
    tx = data.attrs["tm_x"].astype(float).to_numpy()
    ty = data.attrs["tm_y"].astype(float).to_numpy()
    if _is_wgs84(crs):
        lon, lat = tx, ty
    else:
        from pyproj import Transformer
        tf = Transformer.from_crs(crs, _WGS84, always_xy=True)
        lon, lat = tf.transform(tx, ty)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    _check_lonlat(lon, lat, crs)
    return lon, lat


def _fetch_one(lat: float, lon: float, start: str, end: str) -> np.ndarray | None:
    """Daily ET0 (mm) for one location. Prefers AquaScope's OpenMeteoCollector, falls back
    to the Open-Meteo archive API directly. Returns None on failure."""
    try:
        from aquascope.collectors.openmeteo import OpenMeteoCollector
        j = OpenMeteoCollector(mode="weather").fetch_raw(
            latitude=lat, longitude=lon, start_date=start, end_date=end, daily=[_DAILY])
    except Exception:
        try:
            import requests
            j = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon, "start_date": start,
                        "end_date": end, "daily": _DAILY, "timezone": "GMT"},
                timeout=60).json()
        except Exception:
            return None
    series = (j or {}).get("daily", {}).get(_DAILY)
    return np.asarray(series, "float32") if series else None


def et0_for_data(data: GWData, cache_path: str | None = None,
                 retries: int = 4, pause: float = 0.8) -> np.ndarray:
    """Per-well daily ET0 (W, T) aligned to ``data.dates``.

    Loads ``cache_path`` (.npz with et0/well_ids/dates) when it matches this dataset's
    wells and dates; otherwise fetches every well from Open-Meteo, with retries against
    rate limits, then fills any residual gap with the spatially-smooth per-day cross-well
    mean (ET0 varies little over a ~60 km fan). Caches the result if ``cache_path`` is set.
    """
    ids = [str(w) for w in data.well_ids]
    dates = np.array([str(d.date()) for d in data.dates])

    def _coords() -> np.ndarray:
        # WGS84 lat/lon rounded to ~100 m. Computed lazily (it needs pyproj) so an offline
        # cache hit does not require the optional [data] extra.
        lon, lat = wells_lonlat(data)
        return np.round(np.column_stack([lat, lon]), 3).astype("float32")

    if cache_path:
        import os
        if os.path.exists(cache_path):
            c = np.load(cache_path, allow_pickle=True)
            cids = [str(x) for x in c["well_ids"]]
            # ET0 is a function of location, so verify coordinates too -- a cache matching
            # only on ids/dates would silently return the wrong series for a different well
            # set. Legacy caches (no "coords") fall back to id/date match. Short-circuiting
            # keeps _coords() (pyproj) off the offline-cache-hit path.
            if (cids == ids and list(c["dates"]) == list(dates)
                    and ("coords" not in c.files
                         or np.array_equal(c["coords"].astype("float32"), _coords()))):
                return c["et0"].astype("float32")

    lon, lat = wells_lonlat(data)
    coords = _coords()
    start, end = dates[0], dates[-1]
    et0 = np.full((data.n_wells, data.n_days), np.nan, "float32")
    for i in range(data.n_wells):
        for _attempt in range(retries):
            s = _fetch_one(float(lat[i]), float(lon[i]), start, end)
            if s is not None and s.size:
                n = min(data.n_days, s.size)
                et0[i, :n] = s[:n]
                break
            time.sleep(2.0)
        time.sleep(pause)
    # spatial fill for any well/day still missing (rate-limit failures)
    day_mean = np.nanmean(et0, axis=0)
    nan = np.where(np.isnan(et0))
    if nan[0].size:
        et0[nan] = np.take(np.nan_to_num(day_mean), nan[1])
    if cache_path:
        import os
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, et0=et0, well_ids=np.array(ids), dates=dates,
                            coords=coords)
    return et0


def net_recharge(data: GWData, et0: np.ndarray | None = None, coef: float = 1.0,
                 cache_path: str | None = None) -> GWData:
    """A copy of ``data`` with rainfall replaced by net recharge = rain - coef*ET0.

    coef=1.0 (full ET0 subtraction) was selected on the inner 2018 split. The net can be
    negative (an ET-dominated water deficit); the operator's recharge-memory handles it.
    """
    if et0 is None:
        et0 = et0_for_data(data, cache_path)
    net = (np.nan_to_num(data.rainfall) - coef * np.nan_to_num(et0)).astype("float32")
    return replace(data, rainfall=net)

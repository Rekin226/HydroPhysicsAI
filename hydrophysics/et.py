"""Evapotranspiration driver via Open-Meteo FAO-56 ET0 (fetched with AquaScope).

The ``rain - coef*ET0`` driver is a climatic water-balance proxy, not measured aquifer
recharge: infiltration, irrigation, runoff and storage also matter. ET0 is public
ERA5/Open-Meteo
reanalysis (FAO-56 Penman-Monteith, no API key), fetched at each well's WGS84 location and
cached. An earlier 61-well hindcast reported a simulation KGE change from 0.704 to
0.754. That historical comparison does not establish operational forecast or
independent subsidence skill.

Data source: Open-Meteo (https://open-meteo.com), via the AquaScope toolkit
(https://github.com/Rekin226/aquascope) ``OpenMeteoCollector`` — with a direct-API
fallback so this module works without AquaScope installed.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

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
    daily = (j or {}).get("daily", {})
    series, timestamps = daily.get(_DAILY), daily.get("time")
    if not series or not timestamps or len(series) != len(timestamps):
        return None
    try:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        expected = pd.date_range(start, end, freq="D")
        if (index.tz is not None or index.has_duplicates or index.hasnans
                or not index.equals(index.normalize()) or not index.isin(expected).all()):
            return None
        values = pd.Series(np.asarray(series, "float32"), index=index)
        # Keep missing dates in place; never shift later values across a gap.
        values = values.where(np.isfinite(values) & (values >= 0))
        return values.reindex(expected).to_numpy(dtype="float32")
    except (TypeError, ValueError, OverflowError):
        return None


def et0_for_data(data: GWData, cache_path: str | None = None,
                 retries: int = 4, pause: float = 0.8) -> np.ndarray:
    """Per-well daily ET0 (W, T) aligned to ``data.dates``.

    Loads ``cache_path`` (.npz with et0/well_ids/dates) when it matches this dataset's
    wells and dates; otherwise fetches every well from Open-Meteo, with retries against
    rate limits, then fills any residual gap with the spatially-smooth per-day cross-well
    mean. A day with no finite observation at any well raises instead of becoming zero.
    Spatial filling is an imputation assumption; its mask is stored in new caches.
    Caches the result if ``cache_path`` is set.
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
                cached = c["et0"].astype("float32")
                if (cached.shape != (data.n_wells, data.n_days)
                        or not np.isfinite(cached).all() or (cached < 0).any()):
                    raise ValueError("ET0 cache has invalid shape, missing or negative values")
                return cached

    lon, lat = wells_lonlat(data)
    coords = _coords()
    start, end = dates[0], dates[-1]
    et0 = np.full((data.n_wells, data.n_days), np.nan, "float32")
    for i in range(data.n_wells):
        for _attempt in range(retries):
            s = _fetch_one(float(lat[i]), float(lon[i]), start, end)
            if s is not None and s.shape == (data.n_days,) and np.isfinite(s).any():
                et0[i] = np.where(np.isfinite(s) & (s >= 0), s, np.nan)
                break
            time.sleep(2.0)
        time.sleep(pause)
    # spatial fill for any well/day still missing (rate-limit failures)
    available = np.isfinite(et0).sum(axis=0)
    if (available == 0).any():
        missing = dates[available == 0]
        raise ValueError(f"No ET0 observations for {len(missing)} days; first: {missing[0]}. "
                         "Repair the source data before constructing a weather driver.")
    day_mean = np.nansum(et0, axis=0) / available
    imputed = ~np.isfinite(et0)
    nan = np.where(np.isnan(et0))
    if nan[0].size:
        et0[nan] = np.take(day_mean, nan[1])
    if cache_path:
        import os
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, et0=et0, well_ids=np.array(ids), dates=dates,
                            coords=coords, spatially_imputed=imputed)
    return et0


def net_recharge(data: GWData, et0: np.ndarray | None = None, coef: float = 1.0,
                 cache_path: str | None = None) -> GWData:
    """A copy of ``data`` with rainfall replaced by the rain - coef*ET0 proxy.

    coef=1.0 (full ET0 subtraction) was selected on the inner 2018 split. The net can be
    negative (an ET-dominated water deficit); the operator's recharge-memory handles it.
    Missing forcing requires an explicit repair before this transform, not zero fill.
    """
    if et0 is None:
        et0 = et0_for_data(data, cache_path)
    et0 = np.asarray(et0)
    if (et0.shape != data.rainfall.shape or not np.isfinite(et0).all()
            or (et0 < 0).any()):
        raise ValueError("ET0 must be finite, nonnegative and match the rainfall shape")
    if not np.isfinite(data.rainfall).all():
        raise ValueError("Rainfall contains missing values; apply a documented repair first")
    if not np.isfinite(coef) or coef < 0:
        raise ValueError("ET0 coefficient must be finite and nonnegative")
    net = (data.rainfall - coef * et0).astype("float32")
    return replace(data, rainfall=net)

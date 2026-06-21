"""Live recharge-stress monitor: real-time Open-Meteo forcing for the demo.

The plotted groundwater series in this Space are synthetic, so we cannot show a real
*level*. But we CAN show the real forcing the operator consumes: by pulling actual recent
weather (precipitation and FAO-56 ET0) at a station's true coordinates from Open-Meteo's
free public ERA5 archive, computing net recharge (rain - ET0), and running it through the
SAME multi-timescale recharge-memory the PhysicsUDE operator uses, we get a live signal of
whether the aquifer is currently *recharging* or *depleting* relative to that location's
own multi-year climatology.

This is honest: the forcing and the climatology are 100% real public data; only the
groundwater level (which we never claim here) would be synthetic. No API key is required
and the call is cached per station to stay friendly to Open-Meteo's rate limits.

Stdlib-only (urllib) so the Space needs no extra dependency.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np

_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# The operator's longest recharge-memory timescale (~100-day decay): the smooth,
# seasonal-scale accumulation of net recharge. Matches PhysicsUDE(rain_memory=(0.99,...)).
_DECAY = 0.99
_CLIMO_START = "2013-01-01"   # ~13 years of real climatology for the standardization
_ARCHIVE_LAG_DAYS = 6         # the ERA5 archive trails "today" by ~5 days

# Per-station cache: {(round(lat,3), round(lon,3)): result-dict}. Cleared on restart.
_CACHE: dict = {}


def _fetch_archive(lat: float, lon: float, start: str, end: str) -> dict | None:
    """Daily precipitation_sum + ET0 (mm) from the Open-Meteo ERA5 archive. None on error."""
    q = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": start, "end_date": end,
        "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "GMT",
    })
    try:
        with urllib.request.urlopen(f"{_ARCHIVE}?{q}", timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def _recharge_memory(net: np.ndarray, decay: float = _DECAY) -> np.ndarray:
    """Causal IIR accumulation api[t] = decay*api[t-1] + net[t] (the recharge-memory state)."""
    api = np.empty_like(net)
    acc = 0.0
    for t in range(net.size):
        acc = decay * acc + net[t]
        api[t] = acc
    return api


def recharge_stress(lat: float, lon: float, window_days: int = 365) -> dict:
    """Live recharge-stress signal at (lat, lon) from real Open-Meteo weather.

    Returns a dict with:
      ``ok``        - True if the live fetch succeeded.
      ``dates``     - ISO date strings for the last ``window_days``.
      ``z``         - standardized recharge-memory anomaly over that window (sigma units,
                      relative to this location's full real climatology).
      ``net``       - daily net recharge (rain - ET0, mm) over the window.
      ``current``   - latest z value (the "right now" index).
      ``as_of``     - the latest date covered (archive trails ~5 days).
      ``label``     - a human verdict ("recharging" / "near normal" / "depleting").
      ``status``    - message (errors surfaced here when ``ok`` is False).
    """
    key = (round(lat, 3), round(lon, 3))
    if key in _CACHE:
        return _CACHE[key]

    end = date.today() - timedelta(days=_ARCHIVE_LAG_DAYS)
    j = _fetch_archive(lat, lon, _CLIMO_START, end.isoformat())
    daily = (j or {}).get("daily", {})
    times = daily.get("time")
    pr = daily.get("precipitation_sum")
    et = daily.get("et0_fao_evapotranspiration")
    if not times or pr is None or et is None:
        res = {"ok": False, "status": "Live Open-Meteo data is unavailable right now "
               "(rate-limited or offline) — try again in a moment."}
        return res

    pr = np.nan_to_num(np.asarray(pr, float))
    et = np.nan_to_num(np.asarray(et, float))
    net = pr - et                                   # net recharge (mm/day), may be negative
    api = _recharge_memory(net)
    mu, sd = float(api.mean()), float(api.std() + 1e-6)
    z = (api - mu) / sd

    w = min(window_days, z.size)
    cur = float(z[-1])
    label = ("recharging (wetter than normal)" if cur > 0.5
             else "depleting (drought stress)" if cur < -0.5
             else "near normal")
    res = {
        "ok": True,
        "dates": list(times[-w:]),
        "z": z[-w:].astype(float).tolist(),
        "net": net[-w:].astype(float).tolist(),
        "current": cur,
        "as_of": str(times[-1]),
        "label": label,
        "status": f"live · ERA5 archive through {times[-1]}",
    }
    _CACHE[key] = res
    return res

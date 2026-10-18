"""Shared helpers for WP W2 (ET0 extension 2023-26) and WP W4 (rain residual gaps).

Plan: docs/DATA_REMEDIATION_PLAN_2026-10.md, sections "WP W2" and "WP W4".
Code only; data live under results/ and data_fetch_api/ (gitignored).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, L0_NEW, PREREG_DIR, REPO, REPORTS_DIR  # noqa: E402

WP = "W2W4"

# ---- W2 inputs
ET_CACHE = REPO / "results" / "et" / "openmeteo_et0_2012_2022.npz"
GW_STATIONS = REPO / "chou-shui-data" / "data" / "gw_stations.csv"
FORCING_EXP_ET = REPO / "data_fetch_api" / "forcing_experiment" / "et"
OM_L0 = L0_NEW / "openmeteo_et0"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_VAR = "et0_fao_evapotranspiration"
OM_LICENCE = ("CC BY 4.0 (Open-Meteo data licence; attribution 'Weather data by Open-Meteo.com'; "
              "free non-commercial API, 10,000 calls/day, 5,000/hour, 600/minute)")

# ---- W4 inputs
RF_TIMESERIES = REPO / "chou-shui-data" / "data" / "rf_timeseries.csv"
RF_STATIONS = REPO / "chou-shui-data" / "data" / "rf_stations.csv"
CODIS_STATION_LIST = (L0_NEW / "cwa_station_inventory" / "codis_station_list_2026-10-11.json")

# ---- outputs
ET0_PRODUCT = "et0"
RAIN_PRODUCT = "rain_check"


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def points_wgs84():
    """The 61 ET0 points: chou-shui-data gw_stations.csv (st_id, TM_X97/TM_Y97 in EPSG:3826)
    transformed to WGS84 lat/lon, rounded to 5 decimals (about 1 m). These are the same
    locations the calibration cache and the forcing experiment used (st_id keys)."""
    import pandas as pd
    from pyproj import Transformer

    gw = pd.read_csv(GW_STATIONS, encoding="utf-8-sig")
    tf = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(gw["TM_X97"].to_numpy(float), gw["TM_Y97"].to_numpy(float))
    out = pd.DataFrame({"st_id": gw["st_id"].astype(str), "lat": [round(float(v), 5) for v in lat],
                        "lon": [round(float(v), 5) for v in lon],
                        "tm_x": gw["TM_X97"].to_numpy(float), "tm_y": gw["TM_Y97"].to_numpy(float)})
    if len(out) != 61 or out["st_id"].duplicated().any():
        raise ValueError(f"expected 61 unique st_id, got {len(out)}")
    return out


def read_json_untrusted(path: str | os.PathLike) -> dict:
    """Downloaded files are untrusted data: parse as JSON only, never execute."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_new(path: str | os.PathLike, obj, mode: int | None = None) -> Path:
    """Write JSON to a new file (refuses to overwrite)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jsonable)
        f.write("\n")
    if mode is not None:
        os.chmod(p, mode)
    return p


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


__all__ = ["WP", "ET_CACHE", "GW_STATIONS", "FORCING_EXP_ET", "OM_L0", "OM_ARCHIVE", "OM_VAR",
           "OM_LICENCE", "RF_TIMESERIES", "RF_STATIONS", "CODIS_STATION_LIST", "ET0_PRODUCT",
           "RAIN_PRODUCT", "DATAPREP", "PREREG_DIR", "REPORTS_DIR", "REPO", "sha256_file",
           "points_wgs84", "read_json_untrusted", "write_json_new"]

"""WP S3 (v2 code, repair round 2026-10-12; unchanged from v1 apart from the repository
path depth) readers: WRA / NCKU daily heights (L0 vendor cache), NGL tenv3 and NGL step table.

All readers are pure parsers of data files; nothing here executes downloaded content. Run
anything that reads ``data_fetch_api/new_measurements/ngl_gnss`` with ``python -I``.

Units: metres for heights, decimal years for time where noted, pandas ``Timestamp`` (day)
for dates. The WRA cache holds daily ellipsoidal heights (``value``, m); NGL tenv3 holds
daily IGS20 east/north/up offsets from a reference position (``u0 + up`` = height, m).
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]   # v2/ sits one level deeper than v1
LS_CACHE = REPO / "chou-shui-data" / "data" / "ls_cache"
NGL_REVIEW = REPO / "data_fetch_api" / "new_measurements" / "ngl_gnss" / "2026-10-12"
NGL_S3 = REPO / "data_fetch_api" / "new_measurements" / "ngl_gnss" / "2026-10-12_s3"
NGL_S3B = REPO / "data_fetch_api" / "new_measurements" / "ngl_gnss" / "2026-10-12_s3b"

#: WRA meta columns read (never Address or any contact field).
WRA_META_COLS = ["_station_id", "ObservatoryName", "Latitude_4326", "Longitude_4326",
                 "X_3826", "Y_3826", "GroundwaterZoneIdentifier", "CountyCode",
                 "EstablishedTime", "start_datetime_from_dataset", "end_datetime_from_dataset"]

_MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


# ----------------------------------------------------------------------------- WRA / NCKU
def wra_meta() -> pd.DataFrame:
    m = pd.read_parquet(LS_CACHE / "ls-wra-gnss-obs__meta.parquet", columns=WRA_META_COLS)
    return m.rename(columns={"_station_id": "station"}).set_index("station")


def wra_files() -> dict[str, Path]:
    out = {}
    for p in sorted(LS_CACHE.glob("ls-wra-gnss-obs__*.parquet")):
        k = p.name.split("__", 1)[1][:-len(".parquet")]
        if k != "meta":
            out[k] = p
    return out


def read_wra(station: str) -> pd.Series:
    """Daily height (m) of one WRA station from the raw vendor cache (``ls_cache/``)."""
    df = pd.read_parquet(LS_CACHE / f"ls-wra-gnss-obs__{station}.parquet", columns=["value"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    s = s[~s.index.duplicated(keep="first")].sort_index().dropna()
    s.name = station
    return s


def read_wra_vendor_clean_mask(station: str) -> pd.Series | None:
    """True where the vendor's ``ls_cache/clean`` copy blanked a raw value (a check only)."""
    p = LS_CACHE / "clean" / f"ls-wra-gnss-obs__{station}.parquet"
    if not p.is_file():
        return None
    df = pd.read_parquet(p, columns=["value"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return s.isna()


def ncku_meta() -> pd.DataFrame:
    m = pd.read_parquet(LS_CACHE / "ls-ncku-gnss1day-obs__meta.parquet")
    return m.rename(columns={"_station_id": "station"}).set_index("station")


def read_ncku(station: str) -> pd.Series:
    df = pd.read_parquet(LS_CACHE / f"ls-ncku-gnss1day-obs__{station}.parquet",
                         columns=["value"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    s = s[~s.index.duplicated(keep="first")].sort_index().dropna()
    s.name = f"NCKU_{station}"
    return s


# ----------------------------------------------------------------------------- NGL
def ngl_tenv3_path(station: str) -> Path:
    for d in (NGL_S3, NGL_S3B):
        p = d / f"{station}.tenv3"
        if p.is_file():
            return p
    return NGL_REVIEW / f"{station}.tenv3"


def _yymmmdd(s: str) -> pd.Timestamp:
    yy, mon, dd = int(s[:2]), _MON[s[2:5].upper()], int(s[5:7])
    return pd.Timestamp(year=(1900 + yy) if yy >= 90 else (2000 + yy), month=mon, day=dd)


def read_tenv3(path: str | Path) -> pd.DataFrame:
    """NGL tenv3 -> DataFrame indexed by day with columns east, north, up (m, offsets from
    the reference position incl. the integer parts), sig_e/sig_n/sig_u (m), ant (m), lat,
    lon (deg, lon wrapped to [-180, 180)), height (m)."""
    rows = []
    with open(path, encoding="ascii", errors="replace") as f:
        hdr = next(f).split()
        ix = {h: i for i, h in enumerate(hdr)}
        need = ["YYMMMDD", "_e0(m)", "__east(m)", "____n0(m)", "_north(m)", "u0(m)",
                "____up(m)", "_ant(m)", "sig_e(m)", "sig_n(m)", "sig_u(m)",
                "_latitude(deg)", "_longitude(deg)", "__height(m)"]
        miss = [n for n in need if n not in ix]
        if miss:
            raise ValueError(f"{path}: unexpected tenv3 header, missing {miss}")
        for line in f:
            p = line.split()
            if len(p) < len(hdr):
                continue
            try:
                rows.append((_yymmmdd(p[ix["YYMMMDD"]]),
                             float(p[ix["_e0(m)"]]) + float(p[ix["__east(m)"]]),
                             float(p[ix["____n0(m)"]]) + float(p[ix["_north(m)"]]),
                             float(p[ix["u0(m)"]]) + float(p[ix["____up(m)"]]),
                             float(p[ix["sig_e(m)"]]), float(p[ix["sig_n(m)"]]),
                             float(p[ix["sig_u(m)"]]), float(p[ix["_ant(m)"]]),
                             float(p[ix["_latitude(deg)"]]),
                             ((float(p[ix["_longitude(deg)"]]) + 180.0) % 360.0) - 180.0,
                             float(p[ix["__height(m)"]])))
            except (ValueError, KeyError):
                continue
    df = pd.DataFrame(rows, columns=["date", "east", "north", "up", "sig_e", "sig_n", "sig_u",
                                     "ant", "lat", "lon", "height"])
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df


def read_ngl_steps(path: str | Path, stations: set[str]) -> pd.DataFrame:
    """NGL steps.txt rows for ``stations``: code 1 (equipment change, with its description)
    and code 2 (earthquake: threshold distance km, distance km, magnitude, event id)."""
    rows = []
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 3 or p[0] not in stations:
                continue
            try:
                d = _yymmmdd(p[1])
                code = int(p[2])
            except (ValueError, KeyError):
                continue
            if code == 1:
                rows.append({"station": p[0], "date": d, "code": 1,
                             "detail": " ".join(p[3:])})
            elif code == 2 and len(p) >= 7:
                rows.append({"station": p[0], "date": d, "code": 2,
                             "detail": f"eq {p[6]} M{p[5]} dist {float(p[4]):.0f} km "
                                       f"(threshold {float(p[3]):.0f} km)",
                             "eq_mag": float(p[5]), "eq_dist_km": float(p[4]),
                             "eq_thresh_km": float(p[3]), "eq_id": p[6]})
    return pd.DataFrame(rows)


def read_holdings(path: str | Path, stations: set[str]) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 11 or p[0] not in stations:
                continue
            rows.append({"station": p[0], "lat": float(p[1]),
                         "lon": ((float(p[2]) + 180.0) % 360.0) - 180.0,
                         "height_m": float(p[3]), "dtbeg": p[7], "dtend": p[8],
                         "nsol": int(p[10])})
    return pd.DataFrame(rows).set_index("station")


# ----------------------------------------------------------------------------- geometry
def lonlat_to_twd97(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GRS80 Transverse Mercator, central meridian 121E, k0 0.9999, false easting 250 km
    (TWD97 / TM2 zone 121, EPSG:3826). Accurate to centimetres; used only to place stations
    on the 1 km twin grid."""
    a, f = 6378137.0, 1.0 / 298.257222101
    k0, lon0, fe = 0.9999, math.radians(121.0), 250000.0
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = ep2 * np.cos(lat) ** 2
    A = (lon - lon0) * np.cos(lat)
    e4, e6 = e2 * e2, e2 * e2 * e2
    M = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
             - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat)
             + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat)
             - (35 * e6 / 3072) * np.sin(6 * lat))
    x = fe + k0 * n * (A + (1 - t + c) * A ** 3 / 6
                       + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * A ** 5 / 120)
    y = k0 * (M + n * np.tan(lat) * (A * A / 2 + (5 - t + 9 * c + 4 * c * c) * A ** 4 / 24
                                     + (61 - 58 * t + t * t + 600 * c - 330 * ep2)
                                     * A ** 6 / 720))
    return x, y


def grid_cells(xy: np.ndarray, npz) -> np.ndarray:
    """Active-cell index of each point on the twin grid (-1 if the nearest active cell
    centre is farther than one cell size), as ``hydrophysics.twin.mechanism.ring_cells``."""
    mask = np.asarray(npz["mask"])
    dx = float(npz["dx"])
    rows, cols = np.nonzero(mask)
    cent = np.column_stack([float(npz["x0"]) + (cols + 0.5) * dx,
                            float(npz["y0"]) + (rows + 0.5) * dx])
    out = []
    for p in np.atleast_2d(xy):
        d2 = ((cent - p) ** 2).sum(1)
        c = int(np.argmin(d2))
        out.append(c if math.sqrt(d2[c]) <= dx else -1)
    return np.array(out, dtype=int)


def decimal_year(idx: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    y = idx.year.to_numpy()
    start = pd.to_datetime(pd.Series(y).astype(str) + "-01-01").to_numpy()
    end = pd.to_datetime(pd.Series(y + 1).astype(str) + "-01-01").to_numpy()
    frac = (idx.to_numpy() - start) / (end - start)
    return y + frac.astype(float)


def today() -> str:
    return dt.date.today().isoformat()

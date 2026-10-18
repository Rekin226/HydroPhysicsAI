"""WP W2 optional check (plan: "compare with station ET0 or pan evaporation from CWA
agricultural stations"): station FAO-56 Penman-Monteith ET0 and Class-A pan evaporation at
CWA station 467270 (Tianzhong, manned, from 2020-04-10) against et0/v<N> at the nearest of
the 61 points, monthly.

Post-hoc diagnostic, added after pre-registration W2.et0_overlap.v2 (not gating; no decision
depends on it). Station data come from the CODIS daily responses already in L0
(data_fetch_api/new_measurements/cwa_codis_rain/raw/467270_<year>.json, fetched 2026-10-11).

Daily FAO-56 (Allen et al. 1998, eqs. 6-48): Tmax/Tmin, ea from the mean dew point, Rs from
GlobalSolarRadiation (MJ m-2 d-1), P from the mean station pressure, u2 from the mean wind
speed assuming a 10 m anemometer (eq. 47; height not verified), G = 0. A daily value needs
every input numeric and unflagged (flag None or 0). Station days are local (UTC+8); the
product uses GMT days, which matters little for monthly means (st1 2022: see validation).

Usage: python station_et0.py --product <et0 product dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from wcommon import CODIS_STATION_LIST, REPO, points_wgs84, read_json_untrusted, write_json_new

RAW = REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / "raw"
STATION = "467270"
Z_WIND = 10.0


def _num(d: dict, key: str, flag: str):
    if not isinstance(d, dict):
        return np.nan
    v, f = d.get(key), d.get(flag)
    if v is None or (f not in (None, 0)):
        return np.nan
    v = float(v)
    return v if v > -90 else np.nan


def load_station() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW.glob(f"{STATION}_*.json")):
        if p.name.endswith(".meta.json"):
            continue
        j = read_json_untrusted(p)
        for x in j["data"][0]["dts"]:
            rows.append({
                "date": pd.Timestamp(x["DataDate"][:10]),
                "tmax": _num(x.get("AirTemperature"), "Maximum", "Maximumf"),
                "tmin": _num(x.get("AirTemperature"), "Minimum", "Minimumf"),
                "td": _num(x.get("DewPointTemperature"), "Mean", "Meanf"),
                "u": _num(x.get("WindSpeed"), "Mean", "Meanf"),
                "p_hpa": _num(x.get("StationPressure"), "Mean", "Meanf"),
                "rs": _num(x.get("GlobalSolarRadiation"), "Accumulation", "Accumulationf"),
                "pan": _num(x.get("EvaporationClassAPan"), "Accumulation", "Accumulationf")})
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def fao56(df: pd.DataFrame, lat_deg: float, elev_m: float) -> pd.Series:
    tmax, tmin = df["tmax"], df["tmin"]
    tmean = (tmax + tmin) / 2
    es = (0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
          + 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))) / 2
    ea = 0.6108 * np.exp(17.27 * df["td"] / (df["td"] + 237.3))
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    p_kpa = df["p_hpa"] / 10.0
    gamma = 0.000665 * p_kpa
    u2 = df["u"] * 4.87 / math.log(67.8 * Z_WIND - 5.42)
    j = df.index.dayofyear.to_numpy()
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                            + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev_m) * ra
    rs = df["rs"]
    rns = 0.77 * rs
    sigma = 4.903e-9
    ratio = np.clip(rs / rso, 0.25, 1.0)
    rnl = (sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(ea.clip(lower=0))) * (1.35 * ratio - 0.35))
    rn = rns - rnl
    et0 = ((0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea))
           / (delta + gamma * (1 + 0.34 * u2)))
    return et0.where(et0 >= 0)


def monthly(s: pd.Series, min_frac: float = 0.8) -> pd.Series:
    g = s.resample("MS")
    m, c = g.mean(), g.count()
    return m.where(c >= min_frac * m.index.days_in_month)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    a = ap.parse_args(argv)
    d = Path(a.product)
    lst = read_json_untrusted(CODIS_STATION_LIST)
    info = next(it for grp in lst["data"] for it in grp["item"]
                if str(it["stationID"]) == STATION)
    lat, lon, elev = float(info["latitude"]), float(info["longitude"]), float(info["altitude"])
    st = load_station()
    st["et0_pm"] = fao56(st, lat, elev)
    pts = points_wgs84()
    dist = np.hypot((pts["lat"] - lat) * 111.0, (pts["lon"] - lon) * 111.0 * math.cos(
        math.radians(lat)))
    k = int(np.argmin(dist.to_numpy()))
    pid, pkm = str(pts["st_id"].iloc[k]), float(dist.iloc[k])
    z = np.load(d / "et0_daily_2012_2026.npz", allow_pickle=False)
    ids = [str(x) for x in z["well_ids"]]
    days = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    i = ids.index(pid)
    prod = {v: pd.Series(z[key][i].astype(float), index=days)
            for v, key in (("raw_splice", "et0_raw_splice"),
                           ("level_matched", "et0_level_matched"))}
    m = pd.DataFrame({"station_pm": monthly(st["et0_pm"]), "station_pan": monthly(st["pan"]),
                      "om_raw_splice": monthly(prod["raw_splice"]),
                      "om_level_matched": monthly(prod["level_matched"])})
    m = m.loc["2020-05-01":"2026-09-01"]
    m.index.name = "month"
    m.to_csv(d / "station_check_467270_monthly.csv")
    out = {"label": "post-hoc optional diagnostic (not in pre-registration W2.et0_overlap.v2; "
                    "not gating)", "station": STATION, "station_lat": lat, "station_lon": lon,
           "station_altitude_m": elev, "nearest_point": pid, "nearest_point_km": pkm,
           "assumptions": {"anemometer_height_m": Z_WIND, "G": 0, "day_convention_station":
                           "local (UTC+8)", "day_convention_product": "GMT"},
           "daily_valid_pm": int(st["et0_pm"].notna().sum()),
           "daily_valid_pan": int(st["pan"].notna().sum())}
    for period, (p0, p1) in (("2020-05..2022-12 (cache)", ("2020-05-01", "2022-12-01")),
                             ("2023-01..2026-09 (spliced)", ("2023-01-01", "2026-09-01"))):
        s = m.loc[p0:p1]
        res = {}
        for ref in ("station_pm", "station_pan"):
            for v in ("om_raw_splice", "om_level_matched"):
                ok = s[ref].notna() & s[v].notna()
                if ok.sum() < 6:
                    continue
                res[f"{v}_vs_{ref}"] = {
                    "n_months": int(ok.sum()),
                    "mean_ratio_om_over_ref": float((s.loc[ok, v] / s.loc[ok, ref]).mean()),
                    "sum_ratio_om_over_ref": float(s.loc[ok, v].sum() / s.loc[ok, ref].sum()),
                    "bias_mm_d": float((s.loc[ok, v] - s.loc[ok, ref]).mean()),
                    "r": float(np.corrcoef(s.loc[ok, v], s.loc[ok, ref])[0, 1])}
        out[period] = res
    write_json_new(d / "station_check_467270.json", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

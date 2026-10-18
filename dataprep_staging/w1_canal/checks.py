"""Supporting checks of W1 (not gated): head-anomaly partial correlation and crop shares.

Head check (pre-registered in W1_janmay_benchmark_2026-10-12): monthly head anomaly
(month-of-year mean removed) of the Changhua and Yunlin layer-1 and layer-2 well means,
2012-2022, against the 3-month delivery anomaly, partial correlation given the 3-month
fan-mean rain anomaly. Adapted from the scout's head_check.py (same daily median, 31-day
MAD filter and >= 10 days per month; wells with >= 70 % monthly coverage); differences: rain
from the CWA CODIS rebuild (missing days left missing) instead of the zero-filled vendor
cache, sequential reads (CPU budget), and only the metadata columns sid, CountyName and
GroundwaterLayerCode are read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from w1_canal.config import REPO, WELL_DIR, WELL_META

CODIS_RAIN = REPO / "data_fetch_api" / "new_measurements" / "cwa_codis_rain" / \
    "rf_timeseries_codis.csv"
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}


def well_monthly(path, t0="2012-01-01", t1="2022-12-31"):
    w = pd.read_parquet(path, columns=["value"])["value"]
    w = w[(w > -1000) & (w < 1000)][t0:t1]
    if w.empty:
        return None
    dmed = w.resample("D").median()
    roll = dmed.rolling(31, center=True, min_periods=10).median()
    mad = (dmed - roll).abs().rolling(31, center=True, min_periods=10).median()
    dmed = dmed[(dmed - roll).abs() <= 10 * mad.clip(lower=0.05)]
    m = dmed.resample("MS").mean()
    n = dmed.resample("MS").count()
    m[n < 10] = np.nan
    return m


def anom(x: pd.Series) -> pd.Series:
    return x - x.groupby(x.index.month).transform("mean")


def pcorr(x, y, z):
    dd = pd.concat([x, y, z], axis=1).dropna()
    if len(dd) < 24:
        return np.nan, np.nan, len(dd)
    Z = np.c_[np.ones(len(dd)), dd.iloc[:, 2]]
    rx = dd.iloc[:, 0] - Z @ np.linalg.lstsq(Z, dd.iloc[:, 0], rcond=None)[0]
    ry = dd.iloc[:, 1] - Z @ np.linalg.lstsq(Z, dd.iloc[:, 1], rcond=None)[0]
    return (float(np.corrcoef(dd.iloc[:, 0], dd.iloc[:, 1])[0, 1]),
            float(np.corrcoef(rx, ry)[0, 1]), len(dd))


def head_check(series: dict[str, dict[str, pd.Series]], log=print) -> dict:
    """``series[name][county]`` monthly delivery (10^8 m3) on 2012-01..2022-12."""
    idx = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    meta = pd.read_parquet(WELL_META, columns=["sid", "CountyName", "GroundwaterLayerCode"])
    meta = meta[meta.CountyName.isin(COUNTIES.values())]
    H = {}
    for sid in meta.sid:
        p = WELL_DIR / f"{sid}.parquet"
        if p.exists():
            m = well_monthly(p)
            if m is not None:
                H[sid] = m.reindex(idx)
    H = pd.DataFrame(H)
    H = H.loc[:, H.notna().mean() >= 0.7]
    A = H - H.groupby(H.index.month).transform("mean")
    rain = pd.read_csv(CODIS_RAIN, parse_dates=["date time"], index_col="date time")
    rm = rain.mean(axis=1, skipna=True).resample("MS").sum(min_count=20).reindex(idx)
    r3a = anom(rm.rolling(3).sum())
    out = {}
    for county, cn in COUNTIES.items():
        for lay in (1, 2):
            sids = meta[(meta.CountyName == cn) & (pd.to_numeric(
                meta.GroundwaterLayerCode, errors="coerce") == lay)].sid
            cols = [s for s in sids if s in A.columns]
            h = A[cols].mean(axis=1)
            g = {"wells": len(cols)}
            for name, by_county in series.items():
                x = anom(by_county[county].reindex(idx).rolling(3).sum())
                r, pr, n = pcorr(x, h, r3a)
                g[name] = {"r": round(r, 4), "partial_r_given_rain3": round(pr, 4), "n": n}
            out[f"{county}_L{lay}"] = g
    log(f"  head check: {H.shape[1]} wells with >= 70 % coverage")
    return out


PILOT_DIR = REPO / "results" / "dataprep" / "_pilots" / "surface_water_monthly"


def parse_vs_pilot(daily: pd.DataFrame) -> dict:
    """Day-by-day comparison of this parse with the scout's ``choushui_daily_q.csv`` (opened
    through the guard as after-the-fact analysis; the pilot is design-time evidence)."""
    from common import guard

    p = guard.open_product(PILOT_DIR, "after_the_fact_analysis").read_table(
        "choushui_daily_q.csv", columns=["year", "station_no", "month", "day", "q_cms"])
    k = ["year", "station_no", "month", "day"]
    m = daily[k + ["q_cms"]].merge(p, on=k, how="outer", suffixes=("_w1", "_pilot"),
                                   indicator=True)
    both = m[m._merge == "both"]
    same = (np.isclose(both.q_cms_w1, both.q_cms_pilot, rtol=0, atol=1e-9)
            | (both.q_cms_w1.isna() & both.q_cms_pilot.isna()))
    diff = both[~same]
    nan_w1_only = int((diff.q_cms_w1.isna() & diff.q_cms_pilot.notna()).sum())
    nan_pilot_only = int((diff.q_cms_pilot.isna() & diff.q_cms_w1.notna()).sum())
    both_val = diff.dropna(subset=["q_cms_w1", "q_cms_pilot"])
    return {"station_days_w1": int(len(daily)), "station_days_pilot": int(len(p)),
            "only_in_w1": int((m._merge == "left_only").sum()),
            "only_in_pilot": int((m._merge == "right_only").sum()),
            "compared": int(len(both)), "different": int(len(diff)),
            "value_in_w1_nan_in_pilot": nan_pilot_only,
            "nan_in_w1_value_in_pilot": nan_w1_only,
            "both_numeric_different": int(len(both_val)),
            "by_station": {str(s): int(n) for s, n in diff.station_no.value_counts().items()},
            "examples": [{"station": r.station_no, "date": f"{r.year}-{r.month:02d}-{r.day:02d}",
                          "w1": None if pd.isna(r.q_cms_w1) else float(r.q_cms_w1),
                          "pilot": None if pd.isna(r.q_cms_pilot) else float(r.q_cms_pilot)}
                         for r in diff.head(12).itertuples()]}


def janmay_vs_pilot(bench: pd.DataFrame) -> dict:
    """The pdftotext parse of the Jan-May table against the scout's hand transcription."""
    from common import guard

    p = guard.open_product(PILOT_DIR, "after_the_fact_analysis").read_table(
        "jiji_janmay_supply_2015_2021.csv").set_index("year")
    cols = ["chc", "yl", "pub", "ind"]
    d = (bench[cols] - p.reindex(bench.index)[cols]).abs()
    return {"years": [int(y) for y in bench.index], "max_abs_diff_1e8m3": float(d.max().max()),
            "cells_compared": int(d.notna().sum().sum())}


def crop_share_check(monthly: dict[str, np.ndarray], years, moa: pd.DataFrame) -> dict:
    """Crop-1 (Feb-Jun) share of the v4 annual delivery against MOA 691 (2012-2015)."""
    years = list(years)
    out = {}
    for d in ("chc", "yl"):
        x = np.asarray(monthly[d]).reshape(len(years), 12)
        sh = x[:, 1:6].sum(1) / x.sum(1)
        rows = []
        for y in years:
            if y in moa.index and np.isfinite(moa.loc[y, d]):
                rows.append({"year": y, "v4": round(float(sh[years.index(y)]), 4),
                             "moa691": round(float(moa.loc[y, d]), 4)})
        out[d] = {"rows": rows, "v4_mean_all_years": round(float(sh.mean()), 4)}
    return out

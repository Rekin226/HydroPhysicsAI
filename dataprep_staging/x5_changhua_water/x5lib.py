"""WP X5 library: annual non-Jiji volume, monthly timing indicators, the share model, Denton
and Chow-Lin benchmarking. Every rule here is the one declared in X5.changhua_water.v1."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402

from x5_changhua_water import x5const as K  # noqa: E402

MONTHS = pd.period_range("2012-01", "2025-12", freq="M")


# ----------------------------------------------------------------------------- annual volume
def moa_annual() -> pd.DataFrame:
    """MOA 35644 office totals (10^8 m3/yr): crop seasons summed to 2015, '全' from 2016."""
    m = pd.read_csv(K.MOA_CSV, encoding="utf-8-sig",
                    usecols=["處別", "年別", "灌溉用水量_億噸", "期作"])
    m["year"] = m["年別"].astype(int) + 1911
    out = {}
    for k, off in K.MOA_OFFICE.items():
        sub = m[m["處別"].astype(str).str.strip() == off]
        out[k] = sub.groupby("year")["灌溉用水量_億噸"].sum()
        out[k + "_basis"] = sub.groupby("year")["期作"].agg(
            lambda s: "annual" if set(s) == {"全"} else "crop_seasons_summed")
    return pd.DataFrame(out)


def jiji_annual() -> pd.DataFrame:
    j = pd.read_csv(K.JIJI_CSV).set_index("year").sort_index()
    return j[["inflow_1e8_m3", "changhua_irrigation_1e8_m3", "yunlin_irrigation_1e8_m3"]]


def annual_volume() -> tuple[pd.DataFrame, dict]:
    """Changhua non-Jiji water per year with the declared sd and flags."""
    moa, jj = moa_annual(), jiji_annual()
    yl_res = (moa["yl"] - jj["yunlin_irrigation_1e8_m3"]).dropna()
    sd_def = float(yl_res.loc[K.SD_DEF_YEARS].std(ddof=1))
    v = (moa["chc"] - jj["changhua_irrigation_1e8_m3"]).dropna()
    level = float(v.loc[K.LEVEL_YEARS].mean())
    level_sd = float(v.loc[K.LEVEL_YEARS].std(ddof=1))
    rows = []
    for y in K.YEARS_ALL:
        if y in v.index:
            step = abs(float(v.loc[y]) - level) / 2.0 if y <= 2015 else 0.0
            rows.append({"year": y, "moa_chc": float(moa.loc[y, "chc"]),
                         "jiji_chc": float(jj.loc[y, "changhua_irrigation_1e8_m3"]),
                         "value": float(v.loc[y]), "sd": math.hypot(sd_def, step),
                         "sd_def": sd_def, "sd_step": step, "flag": 5,
                         "basis": str(moa.loc[y, "chc_basis"])})
        else:
            rows.append({"year": y, "moa_chc": np.nan,
                         "jiji_chc": float(jj.loc[y, "changhua_irrigation_1e8_m3"]),
                         "value": level, "sd": math.hypot(sd_def, level_sd),
                         "sd_def": sd_def, "sd_step": 0.0, "flag": 6,
                         "basis": "extrapolated: mean of 2016-2022 (MOA year not published)"})
    meta = {"sd_def": sd_def, "level_2016_2022": level, "level_sd_2016_2022": level_sd,
            "yunlin_office_minus_jiji_2016_2022": {int(y): float(yl_res.loc[y])
                                                   for y in K.SD_DEF_YEARS},
            "r_value_vs_inflow_2012_2024": float(np.corrcoef(
                v.loc[2012:2024], jj.loc[2012:2024, "inflow_1e8_m3"])[0, 1])}
    return pd.DataFrame(rows).set_index("year"), meta


# ----------------------------------------------------------------------------- indicators
def coherence_indicator(district: str) -> pd.Series:
    """1 - mean short-pair coherence over the district's canal service cells, per month."""
    g = pd.read_csv(K.WORK / "coherence_monthly.csv")
    g = g[g["district"] == district]
    s = pd.Series(1.0 - g["coh_mean"].to_numpy(), index=pd.PeriodIndex(g["month"], freq="M"))
    return s.reindex(MONTHS)


def river_indicator() -> pd.Series:
    """Monthly mean of max(0, inflow composite - Q(H071 溪州大橋)) in cms, Jan-May only.
    W1 pre-origin/v2 to 2022, W1 full/v4 from 2023."""
    parts = []
    for d, lo, hi in ((K.W1_PRE, "2012-01-01", "2022-12-31"),
                      (K.W1_FULL, "2023-01-01", "2025-12-31")):
        i = pd.read_parquet(d / "inflow_composite_daily.parquet", columns=["date", "value"])
        g = pd.read_parquet(d / "gauges_daily_L1.parquet",
                            columns=["date", "station_no", "value"])
        g = g[g["station_no"] == "H071"]
        m = i.merge(g, on="date", suffixes=("_in", "_xz"))
        m["date"] = pd.to_datetime(m["date"])
        m = m[(m["date"] >= lo) & (m["date"] <= hi)]
        m["d"] = (m["value_in"] - m["value_xz"]).clip(lower=0.0)
        parts.append(m.set_index("date")["d"])
    s = pd.concat(parts).sort_index()
    mon = s.groupby(s.index.to_period("M")).mean().reindex(MONTHS)
    mon[~np.isin(mon.index.month, K.RIVER_MONTHS)] = np.nan
    return mon


def head_indicator(district: str) -> pd.Series:
    """County median over layer-1 wells of the month-to-month change of the monthly head
    (flags 0-2, consecutive months), at least 5 wells; pre-origin rows only (heads are a
    forbidden input of a recorded forcing arm)."""
    p = guard.open_product(K.HEADS, "model_design", window_end="2022-12")
    h = p.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    meta = pd.read_parquet(K.HEADS / "station_meta_preorigin.parquet",
                           columns=["sid", "CountyName", "layer_code"])
    sids = set(meta.loc[(meta["CountyName"] == K.COUNTY[district])
                        & (meta["layer_code"].astype(str) == "1"), "sid"])
    h = h[h["sid"].isin(sids)].copy()
    h["per"] = pd.PeriodIndex(pd.to_datetime(h["date"]), freq="M")
    w = h.pivot_table(index="per", columns="sid", values="value", aggfunc="mean")
    w = w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="M"))
    dh = w.diff()
    n = dh.notna().sum(axis=1)
    med = dh.median(axis=1).where(n >= K.HEAD_MIN_WELLS)
    return med.reindex(MONTHS)


def indicators(district: str) -> pd.DataFrame:
    return pd.DataFrame({"coh": coherence_indicator(district), "riv": river_indicator(),
                         "head": head_indicator(district)}, index=MONTHS)


def standardise(ind: pd.DataFrame, years) -> tuple[pd.DataFrame, dict]:
    """Month-of-year anomaly over the calibration years, divided by the pooled anomaly sd."""
    z = pd.DataFrame(index=ind.index)
    stats = {}
    cal = ind.index.month
    inyr = np.isin(ind.index.year, list(years))
    for c in ind.columns:
        x = ind[c]
        mu = x[inyr].groupby(cal[inyr]).mean()
        a = x - mu.reindex(cal).to_numpy()
        sd = float(a[inyr].std(ddof=1))
        z[c] = a / sd if sd > 0 else np.nan
        stats[c] = {"month_mean": {int(k): float(v) for k, v in mu.items()}, "anomaly_sd": sd,
                    "n_calib_months": int(np.isfinite(a[inyr]).sum())}
    return z, stats


# ----------------------------------------------------------------------------- share model
def ref_shares(vals: pd.Series) -> pd.Series:
    """Monthly share of each year's total (a Series on a monthly PeriodIndex)."""
    tot = vals.groupby(vals.index.year).transform("sum")
    return vals / tot


def calendar(shares: pd.Series, years) -> np.ndarray:
    s = shares[np.isin(shares.index.year, list(years))]
    c = s.groupby(s.index.month).mean().reindex(range(1, 13)).to_numpy()
    return np.nan_to_num(c, nan=0.0)


def predict_shares(cal: np.ndarray, z: pd.DataFrame, b: dict, years) -> pd.Series:
    idx = z.index[np.isin(z.index.year, list(years))]
    zz = z.loc[idx]
    lin = np.zeros(len(idx))
    for k, bk in b.items():
        lin += bk * np.nan_to_num(zz[k].to_numpy(), nan=0.0)
    p = cal[idx.month - 1] * np.exp(lin)
    p = pd.Series(p, index=idx)
    tot = p.groupby(p.index.year).transform("sum")
    return p / tot


def index_series(cal: np.ndarray, z: pd.DataFrame, b: dict, years) -> pd.Series:
    """Un-normalised timing index (the Denton preliminary series)."""
    idx = z.index[np.isin(z.index.year, list(years))]
    lin = np.zeros(len(idx))
    for k, bk in b.items():
        lin += bk * np.nan_to_num(z.loc[idx, k].to_numpy(), nan=0.0)
    return pd.Series(cal[idx.month - 1] * np.exp(lin), index=idx)


def fit_b(cal: np.ndarray, z: pd.DataFrame, shares: pd.Series, keys, years) -> dict:
    keys = list(keys)
    tgt = shares[np.isin(shares.index.year, list(years))]

    def res(v):
        pr = predict_shares(cal, z, dict(zip(keys, v, strict=True)), years)
        return (pr.reindex(tgt.index) - tgt).to_numpy()

    sol = least_squares(res, np.zeros(len(keys)), method="trf")
    return {k: float(x) for k, x in zip(keys, sol.x, strict=True)}


# ----------------------------------------------------------------------------- benchmarking
def _agg(idx: pd.PeriodIndex, years) -> np.ndarray:
    return np.array([[1.0 if p.year == y else 0.0 for p in idx] for y in years])


def denton_pfd(p: pd.Series, annual: pd.Series) -> pd.Series:
    """Proportional first-difference Denton: minimise sum of squared changes of x/p between
    consecutive positive-index months, subject to the annual sums; zero-index months stay 0."""
    years = sorted(set(p.index.year))
    pos = p[p > 0]
    n = len(pos)
    C = _agg(pos.index, years) * pos.to_numpy()[None, :]
    D = np.diff(np.eye(n), axis=0)
    kkt = np.block([[2 * D.T @ D, C.T], [C, np.zeros((len(years), len(years)))]])
    rhs = np.r_[np.zeros(n), annual.reindex(years).to_numpy(dtype=float)]
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    out = pd.Series(0.0, index=p.index)
    out[pos.index] = sol[:n] * pos.to_numpy()
    return out


def chow_lin(p: pd.Series, annual: pd.Series, rhos=None) -> tuple[pd.Series, dict]:
    """Chow-Lin without intercept on the positive-index months, AR(1) rho by grid maximum
    likelihood on the annual aggregates; zero-index months stay 0."""
    years = sorted(set(p.index.year))
    pos = p[p > 0]
    n = len(pos)
    C = _agg(pos.index, years)
    Y = annual.reindex(years).to_numpy(dtype=float)
    x = pos.to_numpy()
    X = C @ x
    best = None
    for rho in (rhos if rhos is not None else np.arange(0.0, 0.951, 0.05)):
        k = np.arange(n)
        S = rho ** np.abs(k[:, None] - k[None, :])
        V = C @ S @ C.T
        Vi = np.linalg.inv(V)
        beta = float((X @ Vi @ Y) / (X @ Vi @ X))
        u = Y - X * beta
        s2 = float(u @ Vi @ u) / len(years)
        ll = -0.5 * (len(years) * math.log(s2) + np.linalg.slogdet(V)[1])
        if best is None or ll > best[0]:
            best = (ll, rho, beta, S, Vi, u)
    ll, rho, beta, S, Vi, u = best
    est = x * beta + S @ C.T @ Vi @ u
    out = pd.Series(0.0, index=p.index)
    out[pos.index] = est
    return out, {"rho": float(rho), "beta": beta, "loglik": float(ll)}


# ----------------------------------------------------------------------------- metrics
def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])

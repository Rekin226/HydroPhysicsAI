"""WP X5 library: InSAR series loading, GNSS anchors, per-epoch anchoring, point sampling,
May-window increments and the T1 metrics. Rules: ``x5const`` (module docstring).

Conventions: InSAR values are vertical-equivalent mm, positive up, relative to the first
epoch (2020-01-12); GNSS heights are converted to mm; an anchored value A(x, t) is the InSAR
value shifted by the per-epoch GNSS offset, relative to the reference epoch. Increments are
positive up (sinking is negative), like the leveling elevation change.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "s4_insar")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_lib as RL  # noqa: E402
import x5const as K  # noqa: E402

MAD = 1.4826


# ----------------------------------------------------------------------------- InSAR
def _segments(gap: np.ndarray | None, shape) -> np.ndarray:
    seg = np.zeros(shape, np.int16)
    if gap is not None:
        seg[1:] = np.cumsum(gap.astype(np.int16), axis=0)
    return seg


def load_p2() -> dict:
    """Run P2 (pre-origin pairs only): vertical-equivalent with the v3 reference (M3)."""
    run, g, v, tab = RL.processed("P2")
    seg = _segments(run["gap"], v.shape)
    valid = run["mask"][None] & np.isfinite(v) & (seg == 0)
    return {"name": "P2", "value": v, "valid": valid, "dates": pd.DatetimeIndex(run["dates"]),
            "x": g["x"], "y": g["y"], "cell": g["cell"], "ref_table": tab}


def load_v3() -> dict:
    """insar_069A/v3 merged multilooked product (F2 + tied second frame)."""
    z = np.load(K.INSAR_ML, allow_pickle=False)
    v = z["value"]
    valid = np.isfinite(v) & ~z["bridged"] & np.isin(z["source"], (1, 2, 3))
    return {"name": "v3", "value": v, "valid": valid,
            "dates": pd.DatetimeIndex(pd.to_datetime(z["dates"])),
            "x": z["x"], "y": z["y"], "cell": z["cell"]}


def point_pixels(ins: dict, xs, ys) -> pd.DataFrame:
    rows = []
    for x, y in zip(xs, ys, strict=True):
        i, j, d = Lb.station_pixel({"x": ins["x"], "y": ins["y"]}, float(x), float(y))
        rows.append({"i": i, "j": j, "dist_m": d})
    return pd.DataFrame(rows)


def point_matrix(ins: dict, xs, ys) -> np.ndarray:
    """(T, P) median of the 3 x 3 valid pixels at each point (NaN: < 4 valid or > 150 m)."""
    px = point_pixels(ins, xs, ys)
    out = np.full((len(ins["dates"]), len(px)), np.nan)
    v, ok, h = ins["value"], ins["valid"], K.PIX_HALF
    for p, r in px.iterrows():
        if r["dist_m"] > K.PIX_MAX_DIST_M:
            continue
        i, j = int(r["i"]), int(r["j"])
        sl = (slice(None), slice(max(i - h, 0), i + h + 1), slice(max(j - h, 0), j + h + 1))
        w = np.where(ok[sl], v[sl], np.nan).reshape(v.shape[0], -1)
        n = np.isfinite(w).sum(1)
        med = _nanmedian(w, 1)
        med[n < K.PIX_MIN_VALID] = np.nan
        out[:, p] = med
    return out


# ----------------------------------------------------------------------------- GNSS
def anchor_stations() -> pd.DataFrame:
    st = pd.read_csv(K.GNSS_STATIONS)
    keep = ((st["family"] == "wra") & st["role"].isin(K.WRA_ROLES)) | (
        (st["family"] == "ngl") & st["role"].isin(K.NGL_ROLES))
    return st.loc[keep, ["family", "station", "x_twd97", "y_twd97", "twin_cell",
                         "role"]].reset_index(drop=True)


def smooth_raw(raw: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """7-day centred median of daily series (mm), the anchors' smoothing."""
    return {s: Lb.smooth_daily(v, K.GNSS_SMOOTH_DAYS) for s, v in raw.items() if len(v) >= 3}


def bench_geometry(months=K.LEV_MONTHS) -> pd.DataFrame:
    """Benchmarks surveyed in every month of ``months``: sid, x, y (EPSG:3826), remark.
    Reads survey presence and positions only (never ``elev_m``)."""
    p = pd.read_parquet(K.LEV_PANEL, columns=["sid", "datetime", "x_3826", "y_3826"])
    p["ym"] = p["datetime"].dt.strftime("%Y-%m")
    sets = [set(p.loc[p["ym"] == m, "sid"]) for m in months]
    both = sorted(set.intersection(*sets))
    g = (p[p["sid"].isin(both) & p["ym"].isin(months)].sort_values(["sid", "datetime"])
         .groupby("sid")[["x_3826", "y_3826"]].last())
    meta = pd.read_parquet(K.LEV_META, columns=["sid", "備註"]).drop_duplicates("sid")
    g = g.join(meta.set_index("sid")).rename(columns={"x_3826": "x", "y_3826": "y",
                                                      "備註": "remark"})
    g["remark"] = g["remark"].fillna("")
    return g.reset_index()


def county_of_points(x, y) -> np.ndarray:
    """'changhua' / 'rest' by the twin cell (zone_county_map v2), 'offgrid' outside."""
    z = np.load(K.COUNTY_MAP, allow_pickle=False)
    mask = z["mask"].astype(bool)
    idx = Lb.cell_index(np.asarray(x, float), np.asarray(y, float), mask, float(z["dx"]),
                        float(z["x0"]), float(z["y0"]))
    ch = z["changhua"].astype(bool)
    return np.where(idx < 0, "offgrid", np.where(ch[np.maximum(idx, 0)], "changhua", "rest"))


def gnss_daily(stations: pd.DataFrame, last_day: str | None = None) -> dict[str, pd.Series]:
    """Daily heights (mm) of the anchor stations, flags 0-1, 7-day centred median."""
    out = {}
    for fam, path in (("wra", K.GNSS_WRA), ("ngl", K.GNSS_NGL)):
        names = set(stations.loc[stations["family"] == fam, "station"])
        if not names:
            continue
        df = pd.read_parquet(path, columns=["station", "date", "value", "flag"])
        df = df[df["station"].isin(names) & df["flag"].isin(K.GNSS_FLAGS)]
        if last_day is not None:
            df = df[df["date"] <= last_day]
        for s, d in df.groupby("station"):
            ser = pd.Series(d["value"].to_numpy() * 1000.0,
                            index=pd.DatetimeIndex(pd.to_datetime(d["date"]))).sort_index()
            ser = ser[~ser.index.duplicated(keep="first")].dropna()
            if len(ser) >= 3:
                out[s] = Lb.smooth_daily(ser, K.GNSS_SMOOTH_DAYS)
    return out


def at_epochs(series: dict[str, pd.Series], names, dates) -> np.ndarray:
    G = np.full((len(dates), len(names)), np.nan)
    for k, s in enumerate(names):
        if s in series:
            G[:, k] = Lb.at_epochs(series[s], pd.DatetimeIndex(dates),
                                   K.GNSS_MATCH_TOL_D).to_numpy()
    return G


# ----------------------------------------------------------------------------- anchoring
def _nanmedian(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def median_polish(D: np.ndarray, min_st: int = K.MIN_STATIONS_EPOCH,
                  sweeps: int = K.POLISH_SWEEPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D (T, S) = k_s + c(t) + e. Returns (c (T,), k (S,), n stations per epoch (T,));
    c is NaN at epochs with fewer than ``min_st`` stations."""
    D = np.asarray(D, float)
    n = np.isfinite(D).sum(1)
    use = n >= min_st
    k = np.zeros(D.shape[1])
    c = np.full(D.shape[0], np.nan)
    for _ in range(sweeps):
        c = np.where(use, _nanmedian(D - k[None], 1), np.nan)
        k = _nanmedian(D - c[:, None], 0)
        k = np.nan_to_num(k)
    return c, k, n


def epoch_corrections(I_st: np.ndarray, G_st: np.ndarray, sx, sy, method: str = "offset"
                      ) -> dict:
    """Per-epoch anchoring terms. Returns {"c": (T,), "a": (T,), "b": (T,), "xc", "yc",
    "k": (S,), "n": (T,), "ref": index of the reference epoch}; the correction at (x, y) is
    c + a (x - xc) / 1000 + b (y - yc) / 1000 minus its value at the reference epoch."""
    D = G_st - I_st
    c, k, n = median_polish(D)
    T = D.shape[0]
    a = np.zeros(T)
    b = np.zeros(T)
    sx, sy = np.asarray(sx, float), np.asarray(sy, float)
    xc, yc = float(np.mean(sx)), float(np.mean(sy))
    if method == "plane":
        R = D - k[None]
        X = np.stack([np.ones_like(sx), (sx - xc) / 1000.0, (sy - yc) / 1000.0], 1)
        for t in range(T):
            ok = np.isfinite(R[t])
            if ok.sum() >= K.MIN_STATIONS_PLANE:
                beta, *_ = Lb.robust_lstsq(X[ok], R[t, ok])
                c[t], a[t], b[t] = beta
            else:
                c[t] = np.nan
    elif method != "offset":
        raise ValueError(method)
    ref = int(np.flatnonzero(np.isfinite(c))[0])
    return {"c": c, "a": a, "b": b, "xc": xc, "yc": yc, "k": k, "n": n, "ref": ref,
            "method": method}


def correction_at(corr: dict, x, y) -> np.ndarray:
    """(T, P) correction at points (x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (corr["c"][:, None] + corr["a"][:, None] * (x[None] - corr["xc"]) / 1000.0
         + corr["b"][:, None] * (y[None] - corr["yc"]) / 1000.0)
    return f - f[corr["ref"]][None]


def station_inputs(ins: dict, stations: pd.DataFrame, series: dict) -> dict:
    names = list(stations["station"])
    I_st = point_matrix(ins, stations["x_twd97"], stations["y_twd97"])
    G_st = at_epochs(series, names, ins["dates"])
    return {"names": names, "I": I_st, "G": G_st, "x": stations["x_twd97"].to_numpy(float),
            "y": stations["y_twd97"].to_numpy(float)}


# ----------------------------------------------------------------------------- May windows
def may_window(dates: pd.DatetimeIndex, year: int) -> np.ndarray:
    c = pd.Timestamp(f"{year}-{K.MAY_CENTRE}")
    lo, hi = c - pd.Timedelta(days=K.MAY_HALF_D), c + pd.Timedelta(days=K.MAY_HALF_D)
    return np.asarray((dates >= lo) & (dates <= hi))


def window_values(A: np.ndarray, dates: pd.DatetimeIndex, year: int):
    """Mean anchored value, mean epoch date (days since 1970) and n epochs per point."""
    sel = may_window(dates, year)
    W = A[sel]
    ok = np.isfinite(W)
    n = ok.sum(0)
    days = (dates[sel] - pd.Timestamp("1970-01-01")).days.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(n > 0, np.nansum(np.where(ok, W, 0.0), 0) / np.maximum(n, 1), np.nan)
        dd = np.where(n > 0, (ok * days[:, None]).sum(0) / np.maximum(n, 1), np.nan)
    return val, dd, n


def may_increment(A: np.ndarray, dates: pd.DatetimeIndex, y0: int, y1: int,
                  scaled: bool = True) -> dict:
    v0, d0, n0 = window_values(A, dates, y0)
    v1, d1, n1 = window_values(A, dates, y1)
    inc = v1 - v0
    span = d1 - d0
    if scaled:
        with np.errstate(invalid="ignore", divide="ignore"):
            inc = inc * K.NOMINAL_DAYS / span
    return {"inc": inc, "span_days": span, "n0": n0, "n1": n1}


# ----------------------------------------------------------------------------- metrics
def ols(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    return b, float(np.mean(y) - b * np.mean(x))


def theil_sen(x, y, max_pairs: int = 400_000, seed: int = 0) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    ok = x[i] != x[j]
    return float(np.median((y[j][ok] - y[i][ok]) / (x[j][ok] - x[i][ok])))


def deming(x, y) -> float:
    """Orthogonal regression slope (equal error variances)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    return float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))


def core_metrics(x_lev, y_ins) -> dict:
    x, y = np.asarray(x_lev, float), np.asarray(y_ins, float)
    b, a = ols(x, y)
    e = y - x
    return {"n": int(len(x)), "slope_ols_insar_on_leveling": b, "intercept_mm": a,
            "rmse_mm": float(np.sqrt(np.mean(e ** 2))), "bias_mm": float(np.mean(e)),
            "sd_err_mm": float(np.std(e)), "robust_sd_err_mm":
                float(MAD * np.median(np.abs(e - np.median(e)))),
            "r": float(np.corrcoef(x, y)[0, 1]), "lev_sd_mm": float(np.std(x)),
            "lev_mean_mm": float(np.mean(x)), "insar_mean_mm": float(np.mean(y))}


def block_bootstrap(x_lev, y_ins, bx, by, n_boot: int = K.N_BOOT, seed: int = K.SEED_BOOT,
                    block_km: float = K.BLOCK_KM) -> dict:
    """90 % percentile intervals of slope and RMSE resampling spatial blocks."""
    x, y = np.asarray(x_lev, float), np.asarray(y_ins, float)
    key = (np.floor(np.asarray(bx) / (block_km * 1000)).astype(int) * 100000
           + np.floor(np.asarray(by) / (block_km * 1000)).astype(int))
    ub, inv = np.unique(key, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(ub))]
    rng = np.random.default_rng(seed)
    sl, rm = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        b, _ = ols(x[idx], y[idx])
        sl.append(b)
        rm.append(float(np.sqrt(np.mean((y[idx] - x[idx]) ** 2))))
    q = (5, 95)
    return {"n_blocks": len(groups), "block_km": block_km, "n_boot": n_boot, "seed": seed,
            "slope_ci90": [float(v) for v in np.percentile(sl, q)],
            "rmse_ci90_mm": [float(v) for v in np.percentile(rm, q)]}


def t1_verdict(m: dict) -> dict:
    lo, hi = K.T1_SLOPE
    ok_n = m["n"] >= K.T1_MIN_N
    ok_s = lo <= m["slope_ols_insar_on_leveling"] <= hi
    ok_r = m["rmse_mm"] <= K.T1_RMSE_MM
    return {"n_ok": bool(ok_n), "slope_ok": bool(ok_s), "rmse_ok": bool(ok_r),
            "pass": bool(ok_n and ok_s and ok_r)}

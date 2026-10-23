"""Barometric loading: hourly pressure at the wells and barometric efficiency (plan WP H1).

Pressure: Open-Meteo ERA5 ``surface_pressure`` (L0
``data_fetch_api/new_measurements/pressure_hourly/openmeteo_era5_<date>/``), one series per
ERA5 0.25-degree cell; each well takes its nearest cell (wells without coordinates take the
centroid of their labelled township).

Barometric response by regression deconvolution (Rasmussen and Crawford 1997):

    dh(t) = sum_k b_k x_k(t) + e(t),        x_k built from lagged pressure differences dp

on first differences of the screened head (clean samples only; episodes, steps, spikes and
dropouts excluded), over the pre-origin window only (2012-01..2022-12, U3/U7). Pressure is in
metres of water (1 hPa = 0.0102 m). Within every 30-day block a constant (and, for the hourly
designs, the harmonics S1-S4, O1, Q1, M2, N2) is projected out of the response and of every
regressor (Frisch-Waugh), so that pumping daily cycles, earth tides and atmospheric tides do
not enter the coefficients; they are fitted on the remaining (synoptic) pressure variation by
Huber IRLS. The cumulative (step) response at the longest lag is A_K; BE = -A_K for a vented
or compensated sensor (A_K near +(1 - BE) marks an uncompensated absolute sensor). se(A_K) by
a 30-day block bootstrap of the per-block normal equations, multiplied by
``config.BARO_SE_INFLATION`` (calibrated on pre-origin within-epoch split halves).

Designs (``DESIGNS``):

* ``hourly_lag48``  the textbook form: hourly dp at lags 0..48 h, one coefficient each;
* ``hourly_grouped`` the same impulse response with lags 0..6 h free and one coefficient per
  lag group 7-12, 13-24, 25-48 h (10 coefficients instead of 49);
* ``daily_lag2``   daily means, dp_day at lags 0, 1, 2 days (block constants only).

The production design is fixed in ``config.BARO_DESIGN`` (chosen on pre-origin split-half
reliability, ``dev_baro_methods.py``).

Loading removed (flag 1): L(t) = sum_k a_k (p(t - k) - p_ref), with a_k the hourly impulse
response implied by the design and p_ref the cell's 2012-2022 mean; corrected head = head - L.
Accepted only if |A_K| > 3 se (calibrated), A_K within (-1.2, 1.2), and enough data (2000
hourly pairs or the daily equivalent, 12 blocks). With ``config.BARO_EPOCHS`` the response is
fitted separately before and after the 2019 logger change (``pipeline.process_well``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import L0_NEW

from . import config as C

PRESSURE_DIR_GLOB = "pressure_hourly/openmeteo_era5_*"
DESIGNS = ("hourly_lag48", "hourly_grouped", "daily_lag2")
GROUPS = [(7, 12), (13, 24), (25, 48)]


def pressure_dir() -> Path:
    ds = sorted(L0_NEW.glob(PRESSURE_DIR_GLOB))
    if not ds:
        raise FileNotFoundError("no Open-Meteo pressure download")
    return ds[-1]


def cell_id(lat: float, lon: float) -> str:
    return f"{lat:.2f}_{lon:.2f}"


def load_pressure(d: Path | None = None) -> pd.DataFrame:
    """Hourly surface pressure (hPa), index local time (naive, Asia/Taipei), one column per
    ERA5 cell. Raw JSON is parsed as data only (call from ``python -I``)."""
    d = pressure_dir() if d is None else d
    rec = json.loads((d / "SOURCE.json").read_text())
    cols: dict[str, list[pd.Series]] = {}
    for f in rec["files"]:
        if f.get("kind") != "cell":
            continue
        name = f["file"]
        parts = name.split("_")
        cid = f"{float(parts[2]):.2f}_{float(parts[3]):.2f}"
        j = json.loads((d / name).read_text())
        if j.get("utc_offset_seconds") != 28800:
            raise ValueError(f"{name}: unexpected utc offset")
        h = j["hourly"]
        s = pd.Series(pd.to_numeric(pd.Series(h["surface_pressure"]), errors="coerce").to_numpy(),
                      index=pd.to_datetime(h["time"]))
        cols.setdefault(cid, []).append(s)
    out = {}
    for cid, parts in cols.items():
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        s = s.where((s > 850) & (s < 1100))
        out[cid] = s
    P = pd.DataFrame(out).sort_index()
    full = pd.date_range(P.index.min(), P.index.max(), freq="h")
    return P.reindex(full)


def pressure_files(d: Path | None = None) -> list[Path]:
    d = pressure_dir() if d is None else d
    rec = json.loads((d / "SOURCE.json").read_text())
    return [d / "SOURCE.json"] + [d / f["file"] for f in rec["files"] if f.get("kind") == "cell"]


def cells_available(P: pd.DataFrame) -> list[tuple[float, float]]:
    return [tuple(float(x) for x in c.split("_")) for c in P.columns]


def nearest_cell(lat: float, lon: float, cells) -> str:
    c = min(cells, key=lambda q: (q[0] - lat) ** 2 + (q[1] - lon) ** 2)
    return cell_id(*c)


def logger_switch(t: np.ndarray) -> int | None:
    """Time (epoch s) of the logger change: the first sample after which >= 90 % of the next
    1000 sampling intervals are <= 11 minutes (sustained 10-minute logging); None if the
    record never switches (or is 10-minute throughout from its start)."""
    if len(t) < 2000:
        return None
    dt = np.diff(t)
    short = (dt <= 660).astype(float)
    c = np.convolve(short, np.ones(1000) / 1000.0, mode="valid")
    idx = np.flatnonzero(c >= 0.9)
    if not len(idx) or idx[0] == 0:
        return None
    return int(t[idx[0] + 1])


# ----------------------------------------------------------------------------- designs
def _harmonic_cols(th: np.ndarray, harmonics: bool = True) -> np.ndarray:
    cols = [np.ones_like(th)]
    if harmonics:
        for spd in C.BARO_HARMONICS_DEG_H.values():
            w = np.radians(spd) * th
            cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def _lagged(dp: np.ndarray, K: int) -> np.ndarray:
    n = len(dp)
    X = np.full((n, K + 1), np.nan)
    for k in range(K + 1):
        X[k:, k] = dp[:n - k]
    return X


def half_hour(p: pd.Series) -> pd.Series:
    """Pressure at hh:30 (mean of hh:00 and hh+1:00), labelled hh:00 to match hourly means."""
    return (p + p.shift(-1)) / 2.0


def design(h: pd.Series, p_hpa: pd.Series, kind: str):
    """Response/regressors for design ``kind``. ``h``: hourly mean head (NaN = missing),
    labelled hh:00 for the mean over [hh:00, hh+1:00); ``p_hpa``: hourly pressure at hh:00
    (raw ERA5). Returns dict(y, X, th, block, ok, cum, harmonics, dt_h) where ``cum`` maps the
    coefficient vector to the cumulative hourly impulse response ``a_k`` (k = 0..48)."""
    if kind in ("hourly_lag48", "hourly_grouped"):
        idx = h.index
        pm = half_hour(p_hpa).reindex(idx).to_numpy() * C.HPA_TO_M
        hv = h.to_numpy()
        y = np.r_[np.nan, np.diff(hv)]
        dp = np.r_[np.nan, np.diff(pm)]
        XL = _lagged(dp, C.BARO_LAGS_H)
        if kind == "hourly_lag48":
            X = XL
            M = np.eye(C.BARO_LAGS_H + 1)
        else:
            cols = [XL[:, k] for k in range(7)]
            M = np.zeros((C.BARO_LAGS_H + 1, 7 + len(GROUPS)))
            for k in range(7):
                M[k, k] = 1.0
            for g, (a, b) in enumerate(GROUPS):
                cols.append(XL[:, a:b + 1].sum(1))
                M[a:b + 1, 7 + g] = 1.0
            X = np.column_stack(cols)
        th = (idx - pd.Timestamp("2012-01-01")).total_seconds().to_numpy() / 3600.0
        harm, dt_h = True, 1.0
    elif kind == "daily_lag2":
        dd = h.groupby(h.index.floor("D"))
        hd = dd.mean().where(dd.count() >= C.BARO_DAY_MIN_HOURS)
        hd = hd.reindex(pd.date_range(hd.index.min(), hd.index.max(), freq="D"))
        pdly = p_hpa.groupby(p_hpa.index.floor("D")).mean().reindex(hd.index).to_numpy()
        pdly = pdly * C.HPA_TO_M
        y = np.r_[np.nan, np.diff(hd.to_numpy())]
        dp = np.r_[np.nan, np.diff(pdly)]
        X = _lagged(dp, 2)
        # the hourly impulse response implied: a daily lag-j coefficient spread evenly over
        # hours 24j .. 24j + 23 (lags beyond 48 h are dropped; the cumulative is kept).
        M = np.zeros((C.BARO_LAGS_H + 1, 3))
        for j in range(3):
            lo, hi = 24 * j, min(24 * j + 23, C.BARO_LAGS_H)
            M[lo:hi + 1, j] = 1.0 / 24.0 * (24.0 / (hi - lo + 1))
        th = (hd.index - pd.Timestamp("2012-01-01")).total_seconds().to_numpy() / 3600.0
        harm, dt_h = False, 24.0
    else:
        raise ValueError(kind)
    block = np.floor(th / (24.0 * C.BARO_BLOCK_DAYS)).astype(int)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    return {"y": y, "X": X, "th": th, "block": block, "ok": ok, "M": M, "harmonics": harm,
            "dt_h": dt_h}


def _block_normal_eqs(D: dict, w: np.ndarray, min_rows: int):
    """Per-block X'WX and X'Wy after projecting the block nuisance columns out (weighted)."""
    y, X, th, block, ok = D["y"], D["X"], D["th"], D["block"], D["ok"]
    K1 = X.shape[1]
    out = []
    for b in np.unique(block[ok]):
        r = ok & (block == b)
        if r.sum() < min_rows:
            continue
        Z = _harmonic_cols(th[r], D["harmonics"])
        sw = np.sqrt(w[r])
        Q, _ = np.linalg.qr(Z * sw[:, None])
        yw = y[r] * sw
        Xw = X[r] * sw[:, None]
        yr = yw - Q @ (Q.T @ yw)
        Xr = Xw - Q @ (Q.T @ Xw)
        out.append({"b": int(b), "XtX": Xr.T @ Xr, "Xty": Xr.T @ yr, "rows": r, "yr": yr,
                    "Xr": Xr, "sw": sw})
    if not out:
        return [], np.zeros((K1, K1)), np.zeros(K1)
    return out, sum(o["XtX"] for o in out), sum(o["Xty"] for o in out)


def fit_design(D: dict, boot: int = C.BARO_BOOT, seed: int = C.BARO_SEED) -> dict:
    """Huber-IRLS block-projected regression; cumulative response, block-bootstrap se."""
    y, ok = D["y"], D["ok"]
    hourly = D["dt_h"] == 1.0
    min_rows = 3 * 24 if hourly else 10
    min_pairs = C.BARO_MIN_PAIRS if hourly else C.BARO_MIN_PAIRS // 24
    nb_all = int(len(np.unique(D["block"][ok]))) if ok.any() else 0
    res = {"n_pairs": int(ok.sum()), "n_blocks": nb_all}
    if res["n_pairs"] < min_pairs or nb_all < C.BARO_MIN_BLOCKS:
        res.update(status="insufficient_data", accepted=False)
        return res
    w = np.ones(len(y))
    sig = np.nan
    for _ in range(C.BARO_IRLS_ITERS):
        blocks, XtX, Xty = _block_normal_eqs(D, w, min_rows)
        if len(blocks) < C.BARO_MIN_BLOCKS:
            break
        beta = np.linalg.solve(XtX + 1e-12 * np.eye(len(Xty)), Xty)
        e = np.concatenate([(o["yr"] - o["Xr"] @ beta) / np.where(o["sw"] > 0, o["sw"], 1.0)
                            for o in blocks])
        sig = 1.4826 * np.median(np.abs(e - np.median(e)))
        if not np.isfinite(sig) or sig <= 0:
            break
        resid = np.zeros(len(y))
        for o in blocks:
            resid[o["rows"]] = (o["yr"] - o["Xr"] @ beta) / np.where(o["sw"] > 0, o["sw"], 1.0)
        u = np.abs(resid) / (C.BARO_HUBER_C * sig)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
    blocks, XtX, Xty = _block_normal_eqs(D, w, min_rows)
    if len(blocks) < C.BARO_MIN_BLOCKS:
        res.update(status="insufficient_data", accepted=False)
        return res
    beta = np.linalg.solve(XtX + 1e-12 * np.eye(len(Xty)), Xty)
    a = D["M"] @ beta                       # hourly impulse response, lags 0..48
    A = np.cumsum(a)
    AKv = float(A[-1])
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    AK = []
    for _ in range(boot):
        pick = rng.integers(0, nb, nb)
        Mx = sum(blocks[i]["XtX"] for i in pick)
        v = sum(blocks[i]["Xty"] for i in pick)
        try:
            AK.append(float(np.sum(D["M"] @ np.linalg.solve(Mx + 1e-12 * np.eye(len(v)), v))))
        except np.linalg.LinAlgError:
            continue
    se_boot = float(np.std(AK, ddof=1)) if len(AK) > 10 else np.nan
    se = se_boot * C.BARO_SE_INFLATION
    yy = sum(float(o["yr"] @ o["yr"]) for o in blocks)
    num = float(beta @ XtX @ beta)
    if not np.isfinite(se):
        status = "no_bootstrap"
    elif abs(AKv) <= C.BARO_SIG * se:
        status = "not_significant"
    elif not (C.BARO_RANGE[0] < AKv < C.BARO_RANGE[1]):
        status = "out_of_range"
    else:
        status = "accepted"
    res.update({
        "coef": beta.tolist(), "a": a.tolist(), "A": A.tolist(), "A_K": AKv, "se_A_K": se,
        "se_A_K_bootstrap": se_boot,
        "BE_long": -AKv, "BE_lag0": -float(A[0]), "BE_lag6": -float(A[6]),
        "BE_lag24": -float(A[24]), "partial_r2": num / yy if yy > 0 else np.nan,
        "resid_sd_m": float(sig) if np.isfinite(sig) else np.nan, "n_blocks_used": nb,
        "sensor_type_hint": ("compensated_or_vented" if AKv < 0 else
                             ("absolute_uncompensated" if AKv > 0.5 else "ambiguous")),
        "status": status, "accepted": status == "accepted"})
    return res


def fit_be(h: pd.Series, p_hpa: pd.Series, kind: str | None = None, **kw) -> dict:
    """Barometric response of one well with the production design (pre-origin data only)."""
    kind = kind or C.BARO_DESIGN
    if h.dropna().size < 48:
        return {"n_pairs": 0, "n_blocks": 0, "status": "insufficient_data", "accepted": False,
                "design": kind}
    out = fit_design(design(h, p_hpa, kind), **kw)
    out["design"] = kind
    return out


# ----------------------------------------------------------------------------- loading
def loading(a: np.ndarray, p_hpa: pd.Series, p_ref_hpa: float) -> pd.Series:
    """L(t) = sum_k a_k (p(t-k) - p_ref) in metres, on the hourly pressure index (values
    at hh:30, labelled hh:00, like the hourly head means)."""
    ph = half_hour(p_hpa)
    pm = (ph.to_numpy() - p_ref_hpa) * C.HPA_TO_M
    K = len(a) - 1
    L = np.zeros(len(pm))
    for k in range(K + 1):
        L[k:] += a[k] * pm[:len(pm) - k]
    L[:K] = np.nan
    return pd.Series(L, index=p_hpa.index)


def loading_at(t_sec: np.ndarray, L_hourly: pd.Series) -> np.ndarray:
    """Loading interpolated linearly to sample times (values sit at hh:30)."""
    x = (L_hourly.index.values.astype("datetime64[s]").astype(np.int64) + 1800).astype(float)
    yv = L_hourly.to_numpy()
    good = np.isfinite(yv)
    if good.sum() < 2:
        return np.full(len(t_sec), np.nan)
    return np.interp(t_sec.astype(float), x[good], yv[good], left=np.nan, right=np.nan)


def oos_ratio(h: pd.Series, p_hpa: pd.Series, a: np.ndarray) -> dict:
    """Out-of-sample check (reported only): with the fixed hourly impulse response ``a``
    (fitted pre-origin), the share of the block-projected hourly-difference variance left
    after removing the loading, robust (MAD^2) and plain, on the series ``h``."""
    D = design(h, p_hpa, "hourly_lag48")
    y, X, th, block, ok = D["y"], D["X"], D["th"], D["block"], D["ok"]
    if ok.sum() < C.BARO_MIN_PAIRS:
        return {"n_pairs": int(ok.sum())}
    e0, e1 = [], []
    for b in np.unique(block[ok]):
        r = ok & (block == b)
        if r.sum() < 3 * 24:
            continue
        Q, _ = np.linalg.qr(_harmonic_cols(th[r]))
        yr = y[r] - Q @ (Q.T @ y[r])
        Xr = X[r] - Q @ (Q.T @ X[r])
        e0.append(yr)
        e1.append(yr - Xr @ a)
    if not e0:
        return {"n_pairs": 0}
    e0, e1 = np.concatenate(e0), np.concatenate(e1)

    def mad2(e):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return {"n_pairs": int(len(e0)), "ratio_plain": float(np.sum(e1 ** 2) / np.sum(e0 ** 2)),
            "ratio_robust": mad2(e1) / mad2(e0) if mad2(e0) > 0 else np.nan}

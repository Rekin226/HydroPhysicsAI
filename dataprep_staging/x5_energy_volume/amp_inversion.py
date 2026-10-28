"""Estimator (c): m3 per kWh from the diurnal drawdown amplitude, independent of any WRA
volume or electricity-to-volume ratio.

Physics. A pump in a confined aquifer whose rate oscillates as Q1 cos(wt) draws the head
down by s = Q1 / (2 pi T) Re[K0(r sqrt(i w S / T)) e^{iwt}] at distance r (the periodic
Theis solution); K0(x e^{i pi/4}) = ker(x) + i kei(x) with x = r sqrt(w S / T). A pump that
runs a rectangular daily pulse of duty d with mean rate Qbar has a fundamental of amplitude
Q1 = F(d) Qbar, F(d) = 2 sin(pi d) / (pi d). If every kept irrigation pump around a well
pumps c m3 per kWh of its own kept electricity E_i (kWh in month m, D_m days), and a share
s_i of it comes from the well's aquifer (X3 cell shares), the monthly fundamental drawdown
amplitude at the well is

    A1(m) = c * G(m),  G(m) = F(d) | sum_i s_i E_i(m) / (D_m 86400) (ker + i kei)(x_i) |
                                  / (2 pi T)                                     [m per m3/kWh]

with the pumps in phase (the diurnal composite shows one clear minimum hour). The observed
A1 is AMP v1 (AMP_V2/amp.py, f_pump fixed at 1 cpd) averaged over the month, less the
non-pumping 1-cpd background (K1 + P1 earth tide from the fitted O1, and a declared
atmospheric S1 floor) in quadrature. c_w = sum_m A1p(m) / sum_m G(m) over the well's valid
Jan-May months. Electricity enters only through the relative size and timing of each pump's
pumping; the volume scale comes from T, S and the observed amplitude.

All constants are in x5const (declared before any test data were read).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import census as C  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

OMEGA = 2.0 * np.pi / 86400.0                    # rad/s, 1 cpd
DEG_H = {"O1": 13.9430356, "K1": 15.0410686, "P1": 14.9589314, "S1": 15.0, "M2": 28.9841042,
         "S2": 30.0, "N2": 28.4397295, "S3": 45.0}
PUMP_CACHE = K.WORK / "kept_irrigation_pumps_2010_2019.npz"
MONTHS = pd.date_range("2010-01-01", "2019-12-01", freq="MS")


# ----------------------------------------------------------------------------- stations
def stations() -> pd.DataFrame:
    """sid, county, layer (1-4), attr (C/U/...), x, y (TWD97; swapped pairs fixed by
    magnitude: easting < 1e6 < northing)."""
    a = pd.read_parquet(K.STATIONS, columns=["sid", "CountyName", "GroundwaterLayerCode",
                                             "LayerAttribute", "LocationByTWD97",
                                             "WellDepth"])
    xy = a["LocationByTWD97"].astype(str).str.split(expand=True)
    u = pd.to_numeric(xy[0], errors="coerce")
    v = pd.to_numeric(xy[1], errors="coerce")
    a["x"] = np.where(u < 1e6, u, v)
    a["y"] = np.where(u < 1e6, v, u)
    a["layer"] = pd.to_numeric(a["GroundwaterLayerCode"], errors="coerce")
    a["attr"] = a["LayerAttribute"].astype("string").fillna("")
    zh = {v_: k for k, v_ in K.COUNTY_ZH.items()}
    a["county"] = a["CountyName"].map(zh)
    return a[["sid", "county", "layer", "attr", "x", "y", "WellDepth"]]


def county_of_point(x: float, y: float, px: np.ndarray, py: np.ndarray,
                    pc: np.ndarray) -> str:
    """County of a location = county of the census pumps within 1.5 km (majority); the
    station table's county labels are wrong for some wells."""
    d2 = (px - x) ** 2 + (py - y) ** 2
    near = d2 < 1500.0 ** 2
    if not near.any():
        near = d2 <= np.partition(d2, 20)[20]
    vals, cnt = np.unique(pc[near], return_counts=True)
    return str(vals[np.argmax(cnt)])


# ----------------------------------------------------------------------------- pumps
def build_pump_cache() -> Path:
    """Kept irrigation pumps of both counties with coordinates, cell, county and kept kWh
    per month 2010-01..2019-12 (float32). Census only (no head data)."""
    K.WORK.mkdir(parents=True, exist_ok=True)
    t0 = C.tidx("2010-01")
    parts = []
    for cn in K.COUNTIES:
        p = C.load_pumps(cn)
        L = C.load_kwh(cn, len(p))
        E = C.bases(p, L)["kept"][:, t0:t0 + len(MONTHS)]
        sel = ((p["purpose_class"] == "irrigation") & p["kept"].astype(bool)
               & p["x_twd97"].notna() & p["y_twd97"].notna()).to_numpy()
        parts.append({"x": p.loc[sel, "x_twd97"].to_numpy(), "y": p.loc[sel, "y_twd97"].to_numpy(),
                      "cell": p.loc[sel, "cell"].to_numpy(dtype="int64"),
                      "county": np.full(sel.sum(), cn), "E": E[sel].astype("float32")})
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    np.savez_compressed(PUMP_CACHE, months=MONTHS.strftime("%Y-%m").to_numpy(), **out)
    return PUMP_CACHE


def load_pumps() -> dict:
    if not PUMP_CACHE.exists():
        build_pump_cache()
    z = np.load(PUMP_CACHE, allow_pickle=False)
    return {k: z[k] for k in z.files}


def layer_shares(cells: np.ndarray, counties: np.ndarray, layer: int) -> np.ndarray:
    """Share of each pump's volume drawn from aquifer ``layer`` (1-4): X3 cell shares
    (share_L{layer-1}); pumps outside the grid take their county's combined share a{layer}."""
    cs = pd.read_csv(K.X3_CELL_SHARES).set_index("cell")[f"share_L{layer - 1}"]
    cc = pd.read_csv(K.X3_COUNTY_SPLIT).set_index("county")[f"a{layer}"]
    s = pd.Series(cells).map(cs).to_numpy(dtype="float64")
    fill = pd.Series(counties).map(cc).to_numpy(dtype="float64")
    return np.where(np.isfinite(s), s, fill)


# ----------------------------------------------------------------------------- storage
def ss_from_be(be: float) -> float:
    """Jacob (1940): BE = n beta / (alpha + n beta), Ss = rho g (alpha + n beta)."""
    return K.RHO_G * K.POROSITY * K.BETA_W / be


def be_table() -> pd.DataFrame:
    b = pd.read_csv(K.BE_CSV, dtype={"sid": str}, usecols=["sid", "accepted", "BE_long"])
    b = b[(b["accepted"].astype(str) == "True") & b["BE_long"].between(0.02, 0.98)]
    return b.groupby("sid")["BE_long"].median()


def storage(sid: str, bet: pd.Series) -> tuple[float, str]:
    """(S, source): Ss from the well's accepted BE epochs, else the median Ss of every
    accepted BE well; S = Ss x the declared aquifer thickness."""
    if sid in bet.index:
        return ss_from_be(float(bet[sid])) * K.AQ_THICKNESS, "BE(well)"
    return float(np.median([ss_from_be(v) for v in bet.to_numpy()])) * K.AQ_THICKNESS, \
        "BE(median of accepted wells)"


# ----------------------------------------------------------------------------- kernel
def kelvin_k0(x: np.ndarray) -> np.ndarray:
    """K0(x e^{i pi/4}) = ker(x) + i kei(x)."""
    return special.ker(x) + 1j * special.kei(x)


def duty_factor(duty_h: float) -> float:
    d = duty_h / 24.0
    return 2.0 * np.sin(np.pi * d) / (np.pi * d)


def g_monthly(r: np.ndarray, share: np.ndarray, E: np.ndarray, days: np.ndarray, T_m2d: float,
              S: float, duty_h: float) -> np.ndarray:
    """G(m) (metres of 1-cpd amplitude per m3/kWh) for pumps at distances ``r`` (m) with
    aquifer shares ``share`` and kept kWh ``E`` (n_pumps, n_months)."""
    T = T_m2d / 86400.0
    x = np.maximum(r, K.R_MIN) * np.sqrt(OMEGA * S / T)
    k = kelvin_k0(x) * share
    rate = E / (days[None, :] * 86400.0)                 # kWh/s per pump-month
    z = (k[:, None] * rate).sum(axis=0)
    return duty_factor(duty_h) * np.abs(z) / (2.0 * np.pi * T)


# ----------------------------------------------------------------------------- heads
def read_series(sid: str, source: str) -> pd.Series:
    """Raw hourly-or-finer head series (local time) of a well: ``ls_cache`` (2010-2011 test
    window) or ``amp`` (AMP_V2 wells, 2012+)."""
    if source == "ls_cache":
        f = K.LS_CACHE / f"gw__{sid}__raw.parquet"
    else:
        f = K.AMP_WELLS / f"{sid}.parquet"
    d = pd.read_parquet(f, columns=["value"])
    s = d["value"]
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()


def qc_hourly(s: pd.Series, t0: str, t1: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """(detrended hourly x with invalid hours set to 0, valid mask, hourly index) on
    [t0, t1). Sentinels and |x - median| > 15 MAD removed (run_ampg_all rule), runs of
    identical values >= FLAT_HOURS removed, hourly mean, gaps <= 6 h interpolated, 15-day
    centred rolling mean removed."""
    idx = pd.date_range(t0, t1, freq="h", inclusive="left")
    s = s[(s.index >= pd.Timestamp(t0)) & (s.index < pd.Timestamp(t1))]
    s = s[(s > -900.0) & (s < 9000.0)].dropna()
    if len(s) < 24 * 30:
        return np.zeros(len(idx)), np.zeros(len(idx), bool), idx
    med = s.median()
    mad = (s - med).abs().median()
    if mad > 0:
        s = s[(s - med).abs() < 15 * mad]
    h = s.resample("h").mean().reindex(idx)
    # flatlines: identical consecutive hourly values over >= FLAT_HOURS hours
    v = h.to_numpy()
    same = np.r_[False, np.isclose(v[1:], v[:-1], atol=1e-9, rtol=0)]
    grp = np.cumsum(~same)
    run = pd.Series(grp).map(pd.Series(grp).value_counts()).to_numpy()
    flat = (run >= K.FLAT_HOURS) & np.isfinite(v)
    h[flat] = np.nan
    h = h.interpolate(limit=6, limit_area="inside")
    valid = h.notna().to_numpy()
    trend = h.rolling(24 * 15, center=True, min_periods=24 * 5).mean()
    x = (h - trend).to_numpy()
    valid &= np.isfinite(x)
    x = np.where(valid, x, 0.0)
    return x, valid, idx


def amp_envelope(x: np.ndarray) -> np.ndarray:
    """AMP v1 (AMP_V2/amp.py) at a fixed 1 cpd."""
    sys.path.insert(0, str(K.AMP_PY.parent))
    from amp import amp_v1  # noqa: E402  (AMP_V2 code, imported read-only)
    return amp_v1(x, 24.0, f_pump=1.0, bandwidth=K.AMP_BW, trend_cut=K.AMP_TREND_CUT,
                  smooth_days=K.AMP_SMOOTH_DAYS)["amp"]


def monthly_mean(v: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"v": np.where(valid, v, np.nan), "ok": valid}, index=idx)
    g = df.resample("MS")
    out = pd.DataFrame({"value": g["v"].mean(), "valid_frac": g["ok"].mean()})
    return out


def harmonic_fit(x: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex,
                 lines=("O1", "K1", "P1", "S1", "M2", "S2", "N2", "S3")) -> dict:
    """Least-squares amplitudes (m) of the named lines over the valid hours."""
    th = (idx - pd.Timestamp("2000-01-01")).total_seconds().to_numpy() / 3600.0
    th = th[valid]
    cols = []
    for ln in lines:
        w = np.radians(DEG_H[ln]) * th
        cols += [np.cos(w), np.sin(w)]
    A = np.column_stack(cols + [np.ones_like(th)])
    coef, *_ = np.linalg.lstsq(A, x[valid], rcond=None)
    return {ln: float(np.hypot(coef[2 * i], coef[2 * i + 1])) for i, ln in enumerate(lines)}


def monthly_lsq_1cpd(x: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex) -> pd.Series:
    """Alternative extractor: per month, the amplitude of the 1 cpd (S1) line fitted with
    2 and 3 cpd harmonics and a constant."""
    th = (idx - pd.Timestamp("2000-01-01")).total_seconds().to_numpy() / 3600.0
    out = {}
    for m, sel in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")):
        i = sel.to_numpy()
        i = i[valid[i]]
        if len(i) < 24 * 10:
            out[m.to_timestamp()] = np.nan
            continue
        cols = []
        for k in (1, 2, 3):
            w = 2 * np.pi * k * th[i] / 24.0
            cols += [np.cos(w), np.sin(w)]
        A = np.column_stack(cols + [np.ones(len(i))])
        c, *_ = np.linalg.lstsq(A, x[i], rcond=None)
        out[m.to_timestamp()] = float(np.hypot(c[0], c[1]))
    return pd.Series(out)


def background(amp_o1: float) -> float:
    return float(np.hypot(K.K1P1_OVER_O1 * amp_o1, K.ATM_S1_M))


# ----------------------------------------------------------------------------- one well
def well_estimate(x: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex, G: pd.Series,
                  months_sel=K.JANMAY, extractor: str = "amp") -> dict:
    """c_w from a QC'd series and its G(m) (indexed by month start)."""
    if extractor == "amp":
        env = amp_envelope(x)
        mm = monthly_mean(env, valid, idx)
        a1 = mm["value"]
    else:
        mm = monthly_mean(x, valid, idx)
        a1 = monthly_lsq_1cpd(x, valid, idx).reindex(mm.index)
    hf = harmonic_fit(x, valid, idx) if valid.sum() > 24 * 60 else {"O1": np.nan}
    b = background(hf["O1"]) if np.isfinite(hf["O1"]) else np.nan
    a1p = np.sqrt(np.maximum(a1 ** 2 - b ** 2, 0.0))
    ok = (mm["valid_frac"] >= K.MIN_VALID_FRAC) & a1.notna()
    if months_sel is not None:
        ok &= np.isin(mm.index.month, months_sel)
    g = G.reindex(mm.index)
    ok &= g.notna() & (g > 0)
    n = int(ok.sum())
    c = float(a1p[ok].sum() / g[ok].sum()) if n >= K.MIN_JM_MONTHS else np.nan
    return {"c": c, "n_months": n, "A1_mean_m": float(a1[ok].mean()) if n else np.nan,
            "A1p_mean_m": float(a1p[ok].mean()) if n else np.nan,
            "G_mean": float(g[ok].mean()) if n else np.nan, "background_m": b,
            "amp_O1_m": hf["O1"], "valid_hours_frac": float(valid.mean())}


def well_geometry(w: pd.Series, P: dict, layer_override: int | None = None) -> dict:
    """Pumps within R_MAX of station row ``w``: distances, aquifer shares, kWh."""
    d = np.hypot(P["x"] - w["x"], P["y"] - w["y"])
    sel = d <= K.R_MAX
    layer = int(layer_override or w["layer"])
    return {"r": d[sel], "share": layer_shares(P["cell"][sel], P["county"][sel], layer),
            "E": P["E"][sel].astype("float64"), "n_2km": int((d <= K.R_COUNT).sum()),
            "n_4km": int(sel.sum()), "layer": layer}


def g_series(geo: dict, months: pd.DatetimeIndex, T_m2d: float, S: float,
             duty_h: float = K.DUTY_H) -> pd.Series:
    mi = pd.Index(MONTHS).get_indexer(months)
    days = months.days_in_month.to_numpy(dtype="float64")
    G = g_monthly(geo["r"], geo["share"], geo["E"][:, mi], days, T_m2d, S, duty_h)
    return pd.Series(G, index=months)

"""Estimators of X5 route 3: earth-tide and barometric specific storage per screen.

* :func:`astro` - equilibrium-tide arguments V + u (Schureman) and nodal factors f;
* :func:`strain_regressors` - areal-strain regressors eps_A,j(t) of the solid earth tide;
* :func:`fit_tides` - block-projected (15-day nuisance: quadratic trend + S1..S4 pumping
  harmonics) Huber-IRLS fit of the head response gain and phase per constituent, with a
  block-bootstrap se;
* :func:`fit_be` - the production barometric design (``h1_heads_qc.baro``, hourly_grouped)
  on the declared window;
* :func:`ss_tide`, :func:`ss_be` - Ss from either route; :func:`hsieh_T` - Hsieh et al. 1987
  transmissivity from the M2 phase shift;
* :func:`load_hourly` - cleaned hourly heads of one AMP well in the window;
* :func:`synth_well` - a synthetic head series with known tide, barometric and pumping
  forcing (the synthetic recovery check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402
from h1_heads_qc import baro  # noqa: E402

DEG = np.pi / 180.0


# ----------------------------------------------------------------------------- astronomy
def _jd(t_ut: pd.DatetimeIndex) -> np.ndarray:
    ns = t_ut.asi8.astype("float64")
    return ns / 86400e9 + 2440587.5


def astro(t_ut: pd.DatetimeIndex) -> dict:
    """{name: (arg_rad, f)}: Schureman's V + u (Greenwich, radians) and the nodal factor f of
    each constituent of :data:`x5sp_const.TIDES` at UT times ``t_ut``."""
    jd = _jd(t_ut)
    T = (jd - 2451545.0) / 36525.0
    s = 218.3164477 + 481267.88123421 * T
    h = 280.46646 + 36000.76983 * T
    p = 83.3532465 + 4069.0137287 * T
    N = (125.04452 - 1934.136261 * T) * DEG
    ut_h = ((jd - 0.5) % 1.0) * 24.0
    Ts = 180.0 + 15.0 * ut_h
    cN, sN = np.cos(N), np.sin(N)
    fm2, um2 = 1.0 - 0.037 * cN, -2.1 * sN
    fo1, uo1 = 1.009 + 0.187 * cN, 10.8 * sN
    fk1, uk1 = 1.006 + 0.115 * cN, -8.9 * sN
    fk2, uk2 = 1.024 + 0.286 * cN, -17.7 * sN
    one, zero = np.ones_like(T), np.zeros_like(T)
    V = {
        "M2": (2 * Ts - 2 * s + 2 * h, fm2, um2),
        "N2": (2 * Ts - 3 * s + 2 * h + p, fm2, um2),
        "S2": (2 * Ts, one, zero),
        "K2": (2 * Ts + 2 * h, fk2, uk2),
        "O1": (Ts - 2 * s + h + 90.0, fo1, uo1),
        "Q1": (Ts - 3 * s + h + p + 90.0, fo1, uo1),
        "K1": (Ts + h - 90.0, fk1, uk1),
        "P1": (Ts - h + 90.0, one, zero),
    }
    return {k: (((v + u) % 360.0) * DEG, f) for k, (v, f, u) in V.items()}


def strain_amp(name: str, lat_deg: float) -> float:
    """Equilibrium areal-strain amplitude eps_A = (2h - 6l) H g(lat) / R (unitless)."""
    c = K.TIDES[name]
    phi = lat_deg * DEG
    geo = np.cos(phi) ** 2 if c["k"] == 2 else np.sin(2 * phi)
    return (2 * K.LOVE_H - 6 * K.LOVE_L) * c["H"] * geo / K.EARTH_R


def c_nu(nu: float = K.NU) -> float:
    """Free-surface volumetric/areal strain factor (1 - 2 nu) / (1 - nu)."""
    return (1.0 - 2.0 * nu) / (1.0 - nu)


def effective_ut(idx_local: pd.DatetimeIndex, label_shift_min: float) -> pd.DatetimeIndex:
    return (idx_local + pd.Timedelta(minutes=label_shift_min)
            - pd.Timedelta(hours=K.LOCAL_UTC_OFFSET_H))


def strain_regressors(t_ut: pd.DatetimeIndex, lat: float, lon: float, names) -> np.ndarray:
    """Columns [eps_c(j), eps_s(j)] = eps_A0 f cos/sin(V + u + k lon) for each constituent:
    the areal strain of the solid tide at the site split into quadratures."""
    A = astro(t_ut)
    cols = []
    for n in names:
        arg, f = A[n]
        e0 = strain_amp(n, lat) * f
        a = arg + K.TIDES[n]["k"] * lon * DEG
        cols += [e0 * np.cos(a), e0 * np.sin(a)]
    return np.column_stack(cols)


def strain_series(t_ut: pd.DatetimeIndex, lat: float, lon: float, names) -> np.ndarray:
    """eps_A(t), the sum of the constituents' areal strain."""
    X = strain_regressors(t_ut, lat, lon, names)
    return X[:, 0::2].sum(axis=1)


# ----------------------------------------------------------------------------- tide fit
def _nuisance(th_h: np.ndarray, b0: float) -> np.ndarray:
    u = (th_h - b0) / K.TIDE_BLOCK_H - 0.5
    cols = [np.ones_like(u), u, u * u]
    for k in (1, 2, 3, 4):              # local-clock pumping harmonics S1..S4
        w = 2 * np.pi * k * th_h / 24.0
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def _blocks(y, X, th, w, ok):
    blk = np.floor(th / K.TIDE_BLOCK_H).astype(int)
    out = []
    for b in np.unique(blk[ok]):
        r = ok & (blk == b)
        if r.sum() < K.TIDE_BLOCK_MIN_ROWS:
            continue
        sw = np.sqrt(w[r])
        Z = _nuisance(th[r], b * K.TIDE_BLOCK_H) * sw[:, None]
        Q, _ = np.linalg.qr(Z)
        yw, Xw = y[r] * sw, X[r] * sw[:, None]
        yr = yw - Q @ (Q.T @ yw)
        Xr = Xw - Q @ (Q.T @ Xw)
        out.append({"rows": r, "sw": sw, "yr": yr, "Xr": Xr, "XtX": Xr.T @ Xr,
                    "Xty": Xr.T @ yr})
    return out


def _solve(blocks, pick=None):
    pick = range(len(blocks)) if pick is None else pick
    XtX = sum(blocks[i]["XtX"] for i in pick)
    Xty = sum(blocks[i]["Xty"] for i in pick)
    return np.linalg.solve(XtX + 1e-30 * np.eye(len(Xty)), Xty)


def fit_tides(h: pd.Series, lat: float, lon: float, names=K.FIT_CONSTITUENTS,
              label_shift_min: float = K.HOUR_LABEL_SHIFT_MIN, boot: int = K.TIDE_BOOT,
              seed: int = K.TIDE_SEED) -> dict:
    """Head response to the areal strain per constituent. ``h``: hourly heads (m) on a
    regular local-time index, NaN = missing. Returns per constituent the gain |G| (m of head
    per unit areal strain), phase theta (deg; head = |G| eps0 cos(V - theta), 180 = undrained
    in anti-phase), bootstrap se, SNR, head amplitude (m) and the Hsieh phase shift
    eta = 180 - theta (deg, wrapped; negative = lag)."""
    y = h.to_numpy(dtype="float64")
    ok = np.isfinite(y)
    t_ut = effective_ut(h.index, label_shift_min)
    X = strain_regressors(t_ut, lat, lon, names)
    th = (h.index - h.index[0].normalize()).total_seconds().to_numpy() / 3600.0
    res = {"n_rows": int(ok.sum())}
    w = np.ones(len(y))
    blocks = _blocks(y, X, th, w, ok)
    if len(blocks) < K.TIDE_MIN_BLOCKS:
        res.update(status="insufficient_data", n_blocks=len(blocks))
        return res
    for _ in range(K.TIDE_IRLS_ITERS):
        beta = _solve(blocks)
        e = np.concatenate([(o["yr"] - o["Xr"] @ beta) / o["sw"] for o in blocks])
        sig = 1.4826 * np.median(np.abs(e - np.median(e)))
        if not np.isfinite(sig) or sig <= 0:
            break
        resid = np.zeros(len(y))
        for o in blocks:
            resid[o["rows"]] = (o["yr"] - o["Xr"] @ beta) / o["sw"]
        u = np.abs(resid) / (K.TIDE_HUBER_C * sig)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        blocks = _blocks(y, X, th, w, ok)
        if len(blocks) < K.TIDE_MIN_BLOCKS:
            res.update(status="insufficient_data", n_blocks=len(blocks))
            return res
    beta = _solve(blocks)
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    B = []
    for _ in range(boot):
        try:
            B.append(_solve(blocks, rng.integers(0, nb, nb)))
        except np.linalg.LinAlgError:
            continue
    B = np.array(B)
    res.update(status="ok", n_blocks=nb)
    f_mean = {n: float(np.mean(astro(t_ut[ok])[n][1])) for n in names}
    for j, n in enumerate(names):
        a, b = beta[2 * j], beta[2 * j + 1]
        g = float(np.hypot(a, b))
        theta = float(np.degrees(np.arctan2(b, a)) % 360.0)
        gb = np.hypot(B[:, 2 * j], B[:, 2 * j + 1])
        pb = np.arctan2(B[:, 2 * j + 1], B[:, 2 * j])
        R = np.hypot(np.mean(np.cos(pb)), np.mean(np.sin(pb)))
        se_ph = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))
        se_g = float(np.std(gb, ddof=1))
        eta = float(((180.0 - theta) + 180.0) % 360.0 - 180.0)
        res[n] = {"gain": g, "theta_deg": theta, "se_gain": se_g, "se_phase_deg": se_ph,
                  "snr": g / se_g if se_g > 0 else np.nan, "eta_deg": eta,
                  "amp_m": g * strain_amp(n, lat) * f_mean[n]}
    return res


# ----------------------------------------------------------------------------- BE fit
def fit_be(h: pd.Series, p_hpa: pd.Series) -> dict:
    """Production barometric response (hourly_grouped) on this series only."""
    out = baro.fit_be(h, p_hpa.reindex(h.index), "hourly_grouped")
    keep = ("status", "accepted", "A_K", "se_A_K", "BE_long", "BE_lag0", "BE_lag6",
            "BE_lag24", "partial_r2", "n_pairs", "n_blocks", "sensor_type_hint")
    return {k: out.get(k) for k in keep}


# ----------------------------------------------------------------------------- Ss, T
def ss_tide(gain: float, nu: float = K.NU) -> float:
    return c_nu(nu) / gain if gain and gain > 0 else np.nan


def ss_be(be: float, n: float = K.N_POR) -> float:
    return K.RHO_G * n * K.BETA_W / be if be and be > 0 else np.nan


def hsieh_response(T, S, omega, rw=K.WELL_RADIUS_M, rc=K.WELL_RADIUS_M):
    """Hsieh, Bredehoeft & Farr (1987): complex well/aquifer head ratio for a harmonic
    pressure head in a confined aquifer, transmissivity T (m2/s), storativity S, angular
    frequency omega (rad/s). arg < 0 is a lag."""
    from scipy.special import kv
    z = np.sqrt(1j * omega * S / T) * rw
    return 1.0 / (1.0 + 1j * omega * rc ** 2 * kv(0, z) / (2.0 * T * z * kv(1, z)))


def hsieh_T(eta_deg: float, S: float, omega: float) -> float:
    """T (m2/s) whose Hsieh phase equals ``eta_deg`` (< 0); NaN when eta >= 0."""
    if not np.isfinite(eta_deg) or eta_deg >= 0:
        return np.nan
    lo, hi = -12.0, 2.0                  # log10 T bracket
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ph = np.degrees(np.angle(hsieh_response(10 ** mid, S, omega)))
        if ph < eta_deg:                 # more lag than observed: T too small
            lo = mid
        else:
            hi = mid
    return float(10 ** (0.5 * (lo + hi)))


# ----------------------------------------------------------------------------- data
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range(K.WINDOW[0], K.WINDOW[1], freq="h")


def daily_good_days(sid: str, daily: pd.DataFrame) -> set:
    d = daily[(daily["sid"] == sid) & daily["flag"].isin(K.GOOD_DAILY_FLAGS)]
    return set(pd.DatetimeIndex(d["date"]).normalize())


def clean_hourly(raw: pd.Series, good_days: set | None) -> pd.Series:
    """Sentinel and exact-zero removal, L1 v3 daily-flag mask, hourly mean on [hh, hh+1),
    spike mask against the 49-h rolling median; on the full window index."""
    v = raw.where((raw.abs() < 1000.0) & (raw != 0.0))
    if good_days is not None:
        keep = v.index.normalize().isin(list(good_days))
        v = v.where(keep)
    hr = v.resample("h", label="left", closed="left").mean()
    hr = hr.reindex(hourly_index())
    med = hr.rolling(49, center=True, min_periods=12).median()
    return hr.where((hr - med).abs() <= K.SPIKE_DEV_M)


def load_hourly(sid: str, daily: pd.DataFrame) -> pd.Series:
    t = pd.read_parquet(K.WELLS_DIR / f"{sid}.parquet", columns=["value"])
    raw = t["value"].loc[K.WINDOW[0]:K.WINDOW[1]]
    return clean_hourly(raw, daily_good_days(sid, daily))


def load_pressure() -> pd.DataFrame:
    p = pd.read_parquet(K.PRESSURE)
    p = p[(p["date"] >= pd.Timestamp(K.WINDOW[0]) - pd.Timedelta(days=3))
          & (p["date"] <= pd.Timestamp(K.WINDOW[1]) + pd.Timedelta(days=1))]
    return p.pivot(index="date", columns="era5_cell", values="surface_pressure_hpa")


# ----------------------------------------------------------------------------- synthetic
def synth_well(ss_true: float, be_true: float, lat: float, lon: float, p_hpa: pd.Series,
               rng: np.random.Generator, *, pump_amp: float, pump_tau_h: float,
               red_sd: float, seas_amp: float, baro_tau_h: float, ocean_amp_m: float = 0.0,
               ocean_phase_deg: float = 0.0, quantum: float = 0.01) -> pd.Series:
    """Hourly synthetic head (m) on the window: undrained earth-tide response to all eight
    constituents (head = -c_nu eps_A / Ss), barometric loading -BE (p - pbar) with a first-
    order delay, an optional ocean M2 term, irrigation pumping with daily on/off cycles
    (random start, duration, amplitude, off days, seasonal envelope, first-order drawdown),
    a night off-peak pump, an annual cycle, AR(1) red noise, spikes, sentinels and gaps;
    10-minute truth averaged to hourly means on [hh, hh+1) and quantised to 1 cm."""
    from scipy.signal import lfilter
    t10 = pd.date_range(K.WINDOW[0], pd.Timestamp(K.WINDOW[1]) + pd.Timedelta(minutes=50),
                        freq="10min")
    n = len(t10)
    t_ut = effective_ut(t10, 0.0)
    eps = strain_series(t_ut, lat, lon, K.SYNTH_CONSTITUENTS)
    head = -c_nu() * eps / ss_true
    if ocean_amp_m > 0:
        arg, f = astro(t_ut)["M2"]
        head = head + ocean_amp_m * f * np.cos(arg + 2 * lon * DEG - ocean_phase_deg * DEG)
    ph = p_hpa.reindex(pd.date_range(t10[0] - pd.Timedelta(hours=1), t10[-1], freq="h"))
    ph = ph.interpolate(limit_direction="both")
    pp = np.interp(t10.asi8.astype(float), ph.index.asi8.astype(float), ph.to_numpy())
    pm = (pp - pp.mean()) * 0.010197162129779282
    if baro_tau_h > 0:
        a = np.exp(-(10.0 / 60.0) / baro_tau_h)
        pm = lfilter([1 - a], [1, -a], pm, zi=[pm[0] * a])[0]
    head = head - be_true * pm
    day = (t10 - t10[0].normalize()).days.to_numpy()
    hod = t10.hour.to_numpy() + t10.minute.to_numpy() / 60.0
    nd = day.max() + 1
    doy = pd.DatetimeIndex(t10[0].normalize() + pd.to_timedelta(np.arange(nd), "D")).dayofyear
    season = 0.6 + 0.4 * np.cos(2 * np.pi * (doy.to_numpy() - 100) / 365.25) ** 2
    on_day = rng.random(nd) < 0.85 * season
    start = rng.uniform(5.0, 9.0, nd)
    dur = rng.uniform(6.0, 12.0, nd)
    amp = pump_amp * season * rng.lognormal(0.0, 0.4, nd) * on_day
    target = np.where((hod >= start[day]) & (hod < start[day] + dur[day]), -amp[day], 0.0)
    night = ((hod >= 22.5) | (hod < 6.5)) & (rng.random(nd)[day] < 0.5)
    target = target - 0.3 * pump_amp * night
    a = np.exp(-(10.0 / 60.0) / pump_tau_h)
    head = head + lfilter([1 - a], [1, -a], target)
    head = head + seas_amp * np.cos(2 * np.pi * (day - 120) / 365.25)
    phi = 0.995 ** (1 / 6)
    e = rng.normal(0.0, red_sd * np.sqrt(1 - phi ** 2), n)
    head = head + lfilter([1.0], [1.0, -phi], e)
    head = head + rng.normal(0.0, 0.002, n)
    hs = pd.Series(head, index=t10)
    hr = hs.resample("h", label="left", closed="left").mean()
    hr = np.round(hr / quantum) * quantum
    m = len(hr)
    sp = rng.random(m) < 0.001
    hr[sp] = hr[sp] + rng.uniform(3.0, 20.0, sp.sum()) * rng.choice([-1, 1], sp.sum())
    hr[rng.random(m) < 0.0005] = -999998.0
    for _ in range(3):
        g0 = int(rng.integers(0, m - 1000))
        hr.iloc[g0:g0 + int(rng.integers(5 * 24, 40 * 24))] = np.nan
    return hr

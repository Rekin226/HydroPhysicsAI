"""Estimator (c), version 2: two corrections found by the v1 synthetic check
(``X5.energy_volume.synthetic_c.v1``; written from its per-well output, before v2 was
declared and before any 2010-2011 head value was read).

1. **Duty per pump-month from the census.** v1 gave every pump the fundamental factor
   F(9.3 h). A pump that draws close to its rating runs most of the day and has almost no
   diurnal fundamental (F -> 0 as the daily hours -> 24). v2 sets each pump-month's daily
   hours to max(9.3 h, H / D_m), capped at 24 h, with H = kept kWh / (0.746 x rated HP)
   the census full-load hours (WRA's own pumping-time rule); pumps without a rated HP keep
   9.3 h.
2. **In-band noise floor.** v1 subtracted only the K1 + P1 tide and a 2 mm S1 floor, so at
   wells whose pumping signal is a few mm the noise inside the AMP band (random-walk and
   quantisation power, unresolved lines) inflated A1. v2 also subtracts, month by month, the
   AMP v1 envelope measured in two side bands at 0.85 and 1.15 cpd (same bandwidth),
   combined in quadrature: A1p^2 = A1^2 - B^2 - N(m)^2, N^2 = (N_0.85^2 + N_1.15^2) / 2.

Everything else is v1 (``amp_inversion``), unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_energy_volume import amp_inversion as A  # noqa: E402
from x5_energy_volume import census as C  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

CACHE_V2 = K.WORK / "kept_irrigation_pumps_v2_2010_2019.npz"
SIDEBANDS = (0.85, 1.15)


def build_cache() -> Path:
    """Kept irrigation pumps with rated kW (0 = unknown) and kept kWh 2010-01..2019-12."""
    K.WORK.mkdir(parents=True, exist_ok=True)
    t0 = C.tidx("2010-01")
    parts = []
    for cn in K.COUNTIES:
        p = C.load_pumps(cn)
        L = C.load_kwh(cn, len(p))
        E = C.bases(p, L)["kept"][:, t0:t0 + len(A.MONTHS)]
        sel = ((p["purpose_class"] == "irrigation") & p["kept"].astype(bool)
               & p["x_twd97"].notna() & p["y_twd97"].notna()).to_numpy()
        hp = pd.to_numeric(p["pump_hp"], errors="coerce").fillna(0.0).clip(lower=0.0)
        parts.append({"x": p.loc[sel, "x_twd97"].to_numpy(dtype="float64"),
                      "y": p.loc[sel, "y_twd97"].to_numpy(dtype="float64"),
                      "cell": p.loc[sel, "cell"].to_numpy(dtype="int64"),
                      "county": np.full(sel.sum(), cn).astype("U8"),
                      "kw": (K.KW_PER_HP * hp[sel]).to_numpy(dtype="float64"),
                      "E": E[sel].astype("float32")})
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    np.savez_compressed(CACHE_V2, months=A.MONTHS.strftime("%Y-%m").to_numpy().astype("U7"),
                        **out)
    return CACHE_V2


def load_pumps() -> dict:
    if not CACHE_V2.exists():
        build_cache()
    z = np.load(CACHE_V2, allow_pickle=False)
    return {k: z[k] for k in z.files}


def well_geometry(w: pd.Series, P: dict) -> dict:
    g = A.well_geometry(w, P)
    d = np.hypot(P["x"] - w["x"], P["y"] - w["y"])
    g["kw"] = P["kw"][d <= K.R_MAX]
    return g


def duty_hours(E: np.ndarray, kw: np.ndarray, days: np.ndarray) -> np.ndarray:
    """(n_pumps, n_months) daily pumping hours on pumping days."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(kw[:, None] > 0, E / kw[:, None], 0.0) / days[None, :]
    return np.clip(np.maximum(K.DUTY_H, h), K.DUTY_H, 24.0)


def f_duty(h: np.ndarray) -> np.ndarray:
    d = h / 24.0
    return 2.0 * np.sin(np.pi * d) / (np.pi * d)


def g_series(geo: dict, months: pd.DatetimeIndex, T_m2d: float, S: float,
             duty_floor_h: float = K.DUTY_H) -> pd.Series:
    mi = pd.Index(A.MONTHS).get_indexer(months)
    days = months.days_in_month.to_numpy(dtype="float64")
    E = geo["E"][:, mi]
    T = T_m2d / 86400.0
    x = np.maximum(geo["r"], K.R_MIN) * np.sqrt(A.OMEGA * S / T)
    k = A.kelvin_k0(x) * geo["share"]
    h = np.maximum(duty_hours(E, geo["kw"], days), duty_floor_h)
    rate = E / (days[None, :] * 86400.0) * f_duty(h)
    z = (k[:, None] * rate).sum(axis=0)
    return pd.Series(np.abs(z) / (2.0 * np.pi * T), index=months)


def sideband_noise(x: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex) -> pd.Series:
    sys.path.insert(0, str(K.AMP_PY.parent))
    from amp import amp_v1  # noqa: E402  (AMP_V2 code, imported read-only)
    n2 = None
    for f in SIDEBANDS:
        env = amp_v1(x, 24.0, f_pump=f, bandwidth=K.AMP_BW, trend_cut=K.AMP_TREND_CUT,
                     smooth_days=K.AMP_SMOOTH_DAYS)["amp"]
        m = A.monthly_mean(env, valid, idx)["value"]
        n2 = m ** 2 if n2 is None else n2 + m ** 2
    return np.sqrt(n2 / len(SIDEBANDS))


def well_estimate(x: np.ndarray, valid: np.ndarray, idx: pd.DatetimeIndex, G: pd.Series,
                  months_sel=K.JANMAY, extractor: str = "amp", noise: bool = True) -> dict:
    if extractor == "amp":
        env = A.amp_envelope(x)
        mm = A.monthly_mean(env, valid, idx)
        a1 = mm["value"]
    else:
        mm = A.monthly_mean(x, valid, idx)
        a1 = A.monthly_lsq_1cpd(x, valid, idx).reindex(mm.index)
    hf = A.harmonic_fit(x, valid, idx) if valid.sum() > 24 * 60 else {"O1": np.nan}
    b = A.background(hf["O1"]) if np.isfinite(hf["O1"]) else np.nan
    nz = sideband_noise(x, valid, idx).reindex(mm.index) if noise else 0.0 * a1
    a1p = np.sqrt(np.maximum(a1 ** 2 - b ** 2 - nz ** 2, 0.0))
    ok = (mm["valid_frac"] >= K.MIN_VALID_FRAC) & a1.notna() & nz.notna()
    if months_sel is not None:
        ok &= np.isin(mm.index.month, months_sel)
    g = G.reindex(mm.index)
    ok &= g.notna() & (g > 0)
    n = int(ok.sum())
    c = float(a1p[ok].sum() / g[ok].sum()) if n >= K.MIN_JM_MONTHS else np.nan
    return {"c": c, "n_months": n, "A1_mean_m": float(a1[ok].mean()) if n else np.nan,
            "A1p_mean_m": float(a1p[ok].mean()) if n else np.nan,
            "noise_mean_m": float(nz[ok].mean()) if n else np.nan,
            "G_mean": float(g[ok].mean()) if n else np.nan, "background_m": b,
            "amp_O1_m": hf["O1"], "valid_hours_frac": float(valid.mean())}

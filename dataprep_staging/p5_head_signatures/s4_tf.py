"""P5 stage 4: per-well transfer-function models (plan WP P5 step 2), pre-origin only.

    h(t) = d + A_R (theta_R * R)(t) + A_S (theta_S * S)(t)
             + (theta_P * (A_irr P_irr + A_oth P_oth))(t) + e(t)

fitted by bounded least squares (scipy ``least_squares``) to the well's L1 v3 monthly heads
(flag 0/1, 2012-01..2022-12; at least TF_MIN_MONTHS months), with gamma block responses
theta(k) = Theta(k+1) - Theta(k), Theta(t) = P(n, t/a) / P(n, M/a) (regularised lower incomplete
gamma, truncated and renormalised at M = TF_MEMORY_MONTHS), as in pastas' Gamma response.

Stresses (monthly, 2012-01..2022-12; the 36 warm-up months before 2012 are filled with each
series' 2012-2022 calendar-month climatology, a declared simplification):

* R: net recharge proxy = rain - ET0 (mm/month / 100): rain from the nearest of the 26
  CODIS-rebuilt gauges with a valid month (>= 25 valid days; next-nearest of 3, else the
  gauge mean), ET0 from the nearest of W2's 61 points (pre-origin file);
* S: W1 surface_water_v4 preorigin/v2 monthly deliveries of the well's county district
  (Changhua or Yunlin; other counties take Yunlin), divided by its 2012-2022 mean;
* P_irr, P_oth: C0 fan in-grid kWh of the irrigation class and of the other non-domestic
  classes (``s3_local_kwh`` regional series), each divided by its 2012-2022 mean, sharing one
  gamma response (separate gains). The domestic class is left out (possibly non-pumping load,
  task update 2026-10-12).

Outputs per well: gains, shapes, the pumping gain ``tf_pump_gain = -(A_irr + A_oth)`` (m of
drawdown per unit-mean regional pumping of both classes: the relative intensity measure),
standard errors from the Jacobian inflated by the residual AR(1) factor (1+phi)/(1-phi)
(uncalibrated), explained variance and residual lag-1 autocorrelation. Two starts (gamma
shapes 1 and 2), the lower cost kept.

Usage: s4_tf.py WORKDIR
Writes WORKDIR/tf_gains.csv and WORKDIR/tf_inputs.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gammainc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402

MONTHS = pd.period_range(C.PRE_MONTHS[0], C.PRE_MONTHS[1], freq="M")
M = C.TF_MEMORY_MONTHS


def gamma_block(n: float, a: float) -> np.ndarray:
    k = np.arange(M + 1, dtype=float)
    Th = gammainc(n, k / a)
    Th = Th / Th[-1] if Th[-1] > 0 else Th
    return np.diff(Th)                      # length M, sums to 1


def padded(x: np.ndarray) -> np.ndarray:
    """Series with M warm-up months of its own calendar-month climatology prepended."""
    clim = pd.Series(x).groupby(np.arange(len(x)) % 12).mean().to_numpy()
    warm = np.array([clim[(i - M) % 12] for i in range(M)])
    return np.r_[warm, x]


def conv(xp: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(theta * x)(t) for the last len(xp) - M months of the padded series."""
    full = np.convolve(xp, theta)[:len(xp)]
    return full[M:]


def rain_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    R = pd.read_csv(C.RAIN_CSV)
    R["date"] = pd.to_datetime(R["date time"])
    R = R[(R.date >= "2012-01-01") & (R.date < "2023-01-01")]
    R["month"] = R["date"].dt.to_period("M")
    vals = R.drop(columns=["date time", "date"]).groupby("month")
    cnt = vals.count()
    mean = vals.mean()
    dim = pd.Series(cnt.index.days_in_month, index=cnt.index)
    tot = mean.mul(dim, axis=0).where(cnt >= 25)
    G = pd.read_csv(C.RAIN_STATIONS, encoding="utf-8-sig")
    return tot.reindex(MONTHS), G


def et0_monthly():
    p = I.et0()
    z = np.load(p.path("et0_daily_2012_2022_preorigin.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    E = pd.DataFrame(np.asarray(z["et0"], float).T, index=dates)
    E = E.where(np.asarray(z["flag"]).T != 8) if "flag" in z.files else E
    Em = E.groupby(E.index.to_period("M")).sum(min_count=25).reindex(MONTHS)
    return Em, np.asarray(z["coords"], float)       # coords: (lat, lon)


def deliveries():
    p = I.surface_water()
    d = pd.read_csv(p.path("district_monthly.csv"), usecols=["date", "district", "value"])
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    S = d.pivot(index="month", columns="district", values="value").reindex(MONTHS)
    return S / S.mean()


def fit_well(h: np.ndarray, X: dict) -> dict:
    """h: monthly heads (NaN missing), X: padded stresses R, S, Pi, Po."""
    ok = np.isfinite(h)

    def model(p):
        d, AR, nR, aR, AS, nS, aS, Ai, Ao, nP, aP = p
        out = d + AR * conv(X["R"], gamma_block(nR, aR)) + AS * conv(X["S"], gamma_block(nS, aS))
        thP = gamma_block(nP, aP)
        out = out + Ai * conv(X["Pi"], thP) + Ao * conv(X["Po"], thP)
        return out

    def resid(p):
        return (model(p) - h)[ok]

    lo = [-np.inf, -50, C.TF_N_BOUNDS[0], C.TF_A_BOUNDS[0], -50, C.TF_N_BOUNDS[0],
          C.TF_A_BOUNDS[0], -50, -50, C.TF_N_BOUNDS[0], C.TF_A_BOUNDS[0]]
    hi = [np.inf, 50, C.TF_N_BOUNDS[1], C.TF_A_BOUNDS[1], 50, C.TF_N_BOUNDS[1],
          C.TF_A_BOUNDS[1], 50, 50, C.TF_N_BOUNDS[1], C.TF_A_BOUNDS[1]]
    best = None
    for n0 in (1.0, 2.0):
        x0 = [float(np.nanmean(h)), 0.0, n0, 3.0, 0.0, n0, 3.0, 0.0, 0.0, n0, 3.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    e = resid(p)
    hv = h[ok]
    n, k = len(e), len(p)
    s2 = float(e @ e) / max(n - k, 1)
    ee = np.full(len(h), np.nan)
    ee[ok] = e
    es = pd.Series(ee)
    phi = float(es.autocorr(1)) if ok.sum() > 12 else np.nan
    infl = (1 + phi) / (1 - phi) if np.isfinite(phi) and -0.99 < phi < 0.99 else np.nan
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, None) * (infl if np.isfinite(infl) else 1.0))
        cov_g = cov[7:9, 7:9] * (infl if np.isfinite(infl) else 1.0)
        se_pump = float(np.sqrt(max(cov_g.sum(), 0.0)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
        se_pump = np.nan
    names = ["d", "A_R", "n_R", "a_R", "A_S", "n_S", "a_S", "A_irr", "A_oth", "n_P", "a_P"]
    out = {nm: float(v) for nm, v in zip(names, p, strict=True)}
    out.update({"se_" + nm: float(s) for nm, s in zip(names, se, strict=True)})
    out.update({"tf_pump_gain": float(-(p[7] + p[8])), "se_tf_pump_gain": se_pump,
                "evp": float(1 - np.var(e) / np.var(hv)) if np.var(hv) > 0 else np.nan,
                "rmse_m": float(np.sqrt(np.mean(e ** 2))), "resid_ar1": phi,
                "n_months": int(n), "cost": float(best.cost), "nfev": int(best.nfev),
                "status": int(best.status),
                "at_bound": ",".join(nm for nm, v, a, b in zip(names, p, lo, hi, strict=True)
                                     if np.isfinite(a) and (abs(v - a) < 1e-6 or
                                                            abs(v - b) < 1e-6))})
    return out


def main():
    work = Path(sys.argv[1])
    meta = I.station_meta()
    lonlat = pd.read_parquet(I.l1().path("station_meta_clean.parquet"),
                             columns=["sid", "lon", "lat"])
    lonlat["sid"] = lonlat["sid"].astype(str)
    meta = meta.merge(lonlat, on="sid", how="left")
    H = I.monthly_heads()
    H = H[H.flag.isin([0, 1]) & (H.month <= C.ORIGIN)]
    Hp = H.pivot(index="month", columns="sid", values="value").reindex(MONTHS.astype(str))
    rain, G = rain_monthly()
    gid = [c for c in rain.columns]
    gxy = G.set_index("rf_id").loc[gid, ["TM_X97", "TM_Y97"]].to_numpy(float)
    Em, ecoords = et0_monthly()
    S = deliveries()
    reg = pd.read_parquet(work / "regional_kwh.parquet")
    reg["month"] = pd.PeriodIndex(reg["month"], freq="M")
    reg = reg.set_index("month").reindex(MONTHS)
    Pi = (reg["fan_irr"] / reg["fan_irr"].mean()).to_numpy()
    Po = (reg["fan_other"] / reg["fan_other"].mean()).to_numpy()
    rain_mean = rain.mean(axis=1)
    rows = []
    corr_info = {}
    for sid in Hp.columns:
        h = Hp[sid].to_numpy(float)
        if np.isfinite(h).sum() < C.TF_MIN_MONTHS:
            continue
        w = meta[meta.sid == sid]
        if w.empty or not np.isfinite(w.x_twd97.iloc[0]):
            continue
        w = w.iloc[0]
        dg = np.hypot(gxy[:, 0] - w.x_twd97, gxy[:, 1] - w.y_twd97)
        order = np.argsort(dg)[:3]
        rw = rain.iloc[:, order[0]].copy()
        for j in order[1:]:
            rw = rw.fillna(rain.iloc[:, j])
        rw = rw.fillna(rain_mean)
        de = np.hypot(ecoords[:, 0] - w.lat, (ecoords[:, 1] - w.lon) * np.cos(np.radians(23.8)))
        ew = Em.iloc[:, int(np.argmin(de))]
        ew = ew.fillna(Em.mean(axis=1))
        R = ((rw - ew) / 100.0).to_numpy(float)
        dist = "changhua" if w.county == "Changhua" else "yunlin"
        Sw = S[dist].to_numpy(float)
        X = {"R": padded(R), "S": padded(Sw), "Pi": padded(Pi), "Po": padded(Po)}
        if not corr_info:
            cm = np.corrcoef(np.vstack([R, Sw, Pi, Po]))
            corr_info = {"stress_correlation_first_well": {"sid": sid, "order": ["R", "S",
                                                                               "P_irr",
                                                                               "P_oth"],
                                                           "matrix": cm.round(3).tolist()}}
        try:
            out = fit_well(h, X)
        except Exception as ex:  # noqa: BLE001
            out = {"error": repr(ex)}
        out.update({"sid": sid, "county": w.county, "layer": w.layer,
                    "rain_gauge": gid[order[0]], "et0_point": int(np.argmin(de)),
                    "district": dist, "fit_window": f"{C.PRE_MONTHS[0]}..{C.PRE_MONTHS[1]}"})
        rows.append(out)
        print(sid, out.get("tf_pump_gain"), out.get("evp"), flush=True)
    T = pd.DataFrame(rows)
    lead = ["sid", "county", "layer", "n_months", "tf_pump_gain", "se_tf_pump_gain", "A_irr",
            "A_oth", "evp", "rmse_m", "resid_ar1"]
    T = T[[c for c in lead if c in T] + [c for c in T.columns if c not in lead]]
    T.to_csv(work / "tf_gains.csv", index=False)
    info = corr_info | {"n_wells": int(len(T)), "memory_months": M,
                        "warmup": "36 months of each stress's 2012-2022 climatology",
                        "pump_series": "C0 preorigin/v3 fan in-grid kept pumps: irrigation; "
                                       "other non-domestic classes"}
    (work / "tf_inputs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

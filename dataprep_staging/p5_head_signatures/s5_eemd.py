"""P5 stage 5: EEMD/HHT plus EOF on first-differenced heads (plan WP P5 step 3), pre-origin.

Benchmark in the spirit of Tseng, Lin & Yu (2024, J. Hydrology 633), whose method and
numbers were checked at abstract level only (plan): the method below is this WP's own
implementation and is not a reproduction of that paper.

* Series: per well, L1 v3 monthly heads (flag 0/1), 2012-01..2022-12; wells with at least
  EEMD_MIN_MONTHS observed months; missing months filled by linear interpolation of the level
  (edges by the nearest value) before first-differencing (the fills are a decomposition aid,
  never a target, U1/U2); dx(t) = h(t) - h(t-1), 131 values.
* EEMD (Wu & Huang 2009): EEMD_ENSEMBLE members of dx + white noise (sd EEMD_NOISE_SD x sd(dx)),
  each sifted by EMD with a fixed EEMD_SIFT_ITERS sifting iterations per IMF, cubic-spline
  envelopes through the extrema with mirrored end extrema, at most EEMD_MAX_IMFS IMFs; IMFs
  averaged over members (seed C.SEED + well index).
* HHT: analytic signal of each IMF (scipy ``hilbert``); mean period from zero crossings;
  the annual IMF is the one whose mean period is closest to 12 months (8-18 months, else
  none); features: its mean instantaneous amplitude (m/month), its variance share of dx.
* EOF: covariance EOF (SVD) of the centred dx matrix of wells with >= EOF_MIN_COVERAGE of the
  131 differences observed (missing differences set to the well mean, i.e. 0 after
  centring); the leading EOF_N_MODES modes: variance fractions, principal components and
  loadings (m/month per unit PC). Each PC's correlation with the differenced regional
  series (fan irrigation and other non-domestic kWh, fan-mean rain) is reported; the mode
  most correlated with differenced fan irrigation kWh (|r|, regional series only) defines
  the per-well feature ``eof_pump_loading``.

Usage: s5_eemd.py WORKDIR
Writes WORKDIR/eemd_features.csv, WORKDIR/eof_modes.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402

MONTHS = pd.period_range(C.PRE_MONTHS[0], C.PRE_MONTHS[1], freq="M").astype(str)


def extrema(x: np.ndarray):
    d = np.diff(x)
    mx = np.flatnonzero((np.r_[d, 0] < 0) & (np.r_[0, d] > 0))
    mn = np.flatnonzero((np.r_[d, 0] > 0) & (np.r_[0, d] < 0))
    return mx, mn


def envelope(x: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    n = len(x)
    if len(idx) < 2:
        return None
    # mirror the first and last two extrema about the series ends
    left = -idx[:2][::-1]
    right = 2 * (n - 1) - idx[-2:][::-1]
    t = np.r_[left, idx, right].astype(float)
    v = np.r_[x[idx[:2][::-1]], x[idx], x[idx[-2:][::-1]]]
    t, u = np.unique(t, return_index=True)
    return CubicSpline(t, v[u])(np.arange(n, dtype=float))


def emd(x: np.ndarray, max_imfs: int, n_sift: int) -> np.ndarray:
    imfs = np.zeros((max_imfs, len(x)))
    r = x.astype(float).copy()
    for i in range(max_imfs):
        h = r.copy()
        ok = True
        for _ in range(n_sift):
            mx, mn = extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                ok = False
                break
            up, lo = envelope(h, mx), envelope(h, mn)
            if up is None or lo is None:
                ok = False
                break
            h = h - 0.5 * (up + lo)
        if not ok:
            break
        imfs[i] = h
        r = r - h
    return imfs


def eemd(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = float(np.std(x))
    acc = np.zeros((C.EEMD_MAX_IMFS, len(x)))
    for _ in range(C.EEMD_ENSEMBLE):
        acc += emd(x + rng.normal(0.0, C.EEMD_NOISE_SD * sd, len(x)), C.EEMD_MAX_IMFS,
                   C.EEMD_SIFT_ITERS)
    return acc / C.EEMD_ENSEMBLE


def mean_period(imf: np.ndarray) -> float:
    zc = np.flatnonzero(np.diff(np.sign(imf)) != 0)
    return 2.0 * len(imf) / len(zc) if len(zc) else np.inf


def main():
    work = Path(sys.argv[1])
    meta = I.station_meta()
    H = I.monthly_heads()
    H = H[H.flag.isin([0, 1]) & (H.month <= C.ORIGIN)]
    Hp = H.pivot(index="month", columns="sid", values="value").reindex(MONTHS)
    rows = []
    for i, sid in enumerate(Hp.columns):
        h = Hp[sid]
        nobs = int(h.notna().sum())
        if nobs < C.EEMD_MIN_MONTHS:
            continue
        hf = h.interpolate(limit_direction="both").to_numpy(float)
        dx = np.diff(hf)
        imfs = eemd(dx, C.SEED + i)
        per = np.array([mean_period(m) for m in imfs])
        var = imfs.var(axis=1) / np.var(dx)
        amp = np.array([np.mean(np.abs(hilbert(m))) for m in imfs])
        cand = np.flatnonzero((per >= 8) & (per <= 18))
        k = int(cand[np.argmin(np.abs(per[cand] - 12))]) if len(cand) else -1
        row = {"sid": sid, "n_obs_months": nobs, "n_filled_months": int(len(MONTHS) - nobs),
               "sd_dx_m": float(np.std(dx)), "annual_imf": k,
               "eemd_annual_amp_m": float(amp[k]) if k >= 0 else np.nan,
               "eemd_annual_share": float(var[k]) if k >= 0 else np.nan}
        for j in range(C.EEMD_MAX_IMFS):
            row[f"imf{j}_period_months"] = float(per[j])
            row[f"imf{j}_var_share"] = float(var[j])
            row[f"imf{j}_hht_amp_m"] = float(amp[j])
        rows.append(row)
        print(sid, k, row["eemd_annual_amp_m"], flush=True)
    E = pd.DataFrame(rows).merge(meta[["sid", "county", "layer"]], on="sid", how="left")

    # ---------------------------------------------------------------- EOF
    D = Hp.diff().iloc[1:]
    cov = D.notna().mean()
    keep = cov[cov >= C.EOF_MIN_COVERAGE].index
    X = D[keep]
    X = X - X.mean()
    X = X.fillna(0.0).to_numpy(float).T           # wells x months
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    frac = s ** 2 / np.sum(s ** 2)
    reg = pd.read_parquet(work / "regional_kwh.parquet").set_index("month").reindex(MONTHS)
    dreg = reg[["fan_irr", "fan_other"]].diff().iloc[1:]
    R = pd.read_csv(C.RAIN_CSV)
    R["month"] = pd.to_datetime(R["date time"]).dt.to_period("M").astype(str)
    rain = R.drop(columns=["date time"]).groupby("month").mean().mean(axis=1).reindex(MONTHS)
    modes = []
    for j in range(C.EOF_N_MODES):
        pc = Vt[j] * s[j] / np.sqrt(X.shape[0])
        r_irr = float(np.corrcoef(pc, dreg["fan_irr"])[0, 1])
        r_oth = float(np.corrcoef(pc, dreg["fan_other"])[0, 1])
        r_rain = float(np.corrcoef(pc, rain.iloc[1:].to_numpy())[0, 1])
        modes.append({"mode": j, "var_frac": float(frac[j]), "r_d_fan_irr": r_irr,
                      "r_d_fan_other": r_oth, "r_rain": r_rain,
                      "pc": pc.round(6).tolist()})
    jp = int(np.argmax([abs(m["r_d_fan_irr"]) for m in modes]))
    sign = np.sign(modes[jp]["r_d_fan_irr"]) or 1.0
    load = U[:, jp] * np.sqrt(X.shape[0]) * sign    # m/month per unit PC, positive = drawdown-
    #                                                 # with-pumping sign convention flipped below
    L = pd.DataFrame({"sid": list(keep), "eof_pump_loading": -load})
    for j in range(C.EOF_N_MODES):
        L[f"eof{j}_loading"] = U[:, j] * np.sqrt(X.shape[0])
    E = E.merge(L, on="sid", how="outer")
    E.to_csv(work / "eemd_features.csv", index=False)
    info = {"n_wells_eemd": int(E["eemd_annual_amp_m"].notna().sum()),
            "n_wells_eof": int(len(keep)), "pump_mode": jp,
            "pump_mode_sign_note": "eof_pump_loading = -(loading x sign(r(PC, d fan_irr))): "
                                   "positive where heads fall when regional irrigation "
                                   "pumping rises",
            "modes": modes, "months": [MONTHS[1], MONTHS[-1]]}
    (work / "eof_modes.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "modes"}, indent=1))
    print([{k: v for k, v in m.items() if k != "pc"} for m in modes])


if __name__ == "__main__":
    main()

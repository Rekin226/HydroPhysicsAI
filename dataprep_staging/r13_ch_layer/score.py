"""Retrospective held-out-years test of a statistical head layer for Changhua.

Label: post-failure, exploratory, retrospective (user ruling R13.1, 2026-11-12). It follows the
failed Changhua pumping ladder (R1..R4p, R*) and cannot make any claim 'supported': heads of
2012-2022 are a spent unit (they scored every rung, ``I1_changhua_drivers`` and the design
study of ``i1/hybrid_tf``, whose forward-chaining folds already covered 2020-2022). It is run
once; the result file is never overwritten.

Layer  TF  : the transfer-function model of ``i1/hybrid_tf`` with its frozen hyperparameters
             (8 drivers, causal filters of 0, 2, 6 and 18 months, month-of-year dummies,
             ridge alpha 1), one fit per well on head months up to 2019-12, predicting
             2020-01..2022-12 from recorded drivers. No head after 2019-12 enters.
Arm    PR  : TF with every driver recorded.
Arm    ZR  : the same fitted model with both electricity drivers replaced, from 2020-01, by
             their month-of-year mean over the fit months. PR - ZR is the layer's response
             to the recorded electricity anomaly.
Arm    CL  : the well's month-of-year mean over the fit months.
Arm    Q0  : the page model's family without Changhua pumping, its own temporal-holdout fit
             (heads to 2019-12; ``stage3_datum_gate_q0_temporal``).
Arm    R1c : the Changhua pumping model, the same holdout (``stage3_datum_gate_fcz_temporal``).
Unit       : Changhua wells with >= 24 observed fit months and >= 24 observed scored months
             and a prediction from every arm; months 2020-01..2022-12.
Scoring    : every arm and the observations are re-zeroed per well at their mean over the
             well's observed fit months, so a level offset known at the origin costs no arm.
Skill rule : 'layer ahead' iff RMSE(PR) is below RMSE(Q0), RMSE(R1c) and RMSE(CL) and, for
             each of the three, the 10 km spatial-block bootstrap 95 % lower bound of
             MSE(arm) - MSE(PR) is above 0; otherwise 'layer not ahead'.
Response   : slope b of (obs - ZR) on (PR - ZR), both demeaned per well over the scored months
             (N2C's estimator). Read only if the RMS of the demeaned contrast is >= 0.05 m.
             'response supported' iff the bootstrap 95 % lower bound of MSE(ZR) - MSE(PR) is
             above 0 and the 90 % interval of b lies above 0; 'size consistent' iff that
             interval also contains 1; otherwise 'response not supported'.
Report only: Yunlin; a placebo in which both electricity drivers of a Changhua well are
             Yunlin's county electricity; the layer's linear head change under a sustained
             30 % cut of both electricity drivers (first-year mean and steady state), which
             lies outside the anomalies the fit has seen.
Disclosed  : the physics arms were fitted on the same heads to 2019-12, with forcing recorded
             in the scored months, as the layer; the layer's hyperparameters were chosen on
             2012-2022 heads; electricity, canal water and rain co-vary in drought years, so
             the electricity term is a conditional association and no causal lever.

    python -m dataprep_staging.r13_ch_layer.score --check     # counts only, no score
    python -m dataprep_staging.r13_ch_layer.score             # scored once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FIT_END = "2019-12"
BLOCK_M = 10_000.0
SEED = 20261115
N_BOOT = 4000
MIN_FIT, MIN_SCORED = 24, 24
INFORMATIVE_M = 0.05
CUT = 0.30
ELEC = ("electricity_local_3km", "electricity_county")
PHYS = {"Q0": "stage3_datum_gate_q0_temporal", "R1c": "stage3_datum_gate_fcz_temporal"}
RESULTS = os.environ.get("HYDRO_RESULTS", "results")
OUT = os.path.join(RESULTS, "dataprep", "reports", "CHL", "retro_2020_2022")


def rezero(a: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """(W, T) minus each row's mean over its own ``fit`` months (NaN-safe)."""
    m = np.where(fit, a, np.nan)
    return a - np.nanmean(m, axis=1, keepdims=True)


def block_ids(xy: np.ndarray) -> np.ndarray:
    _, g = np.unique(np.floor(xy / BLOCK_M).astype(int), axis=0, return_inverse=True)
    return np.asarray(g).ravel()


def boot_mean(d: np.ndarray, g: np.ndarray, rng) -> tuple[float, float]:
    """5 % and 95 % quantiles of the block-bootstrap mean of ``d`` (blocks ``g``)."""
    sums = np.bincount(g, weights=d)
    cnts = np.bincount(g).astype(float)
    ix = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    m = sums[ix].sum(1) / cnts[ix].sum(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def slope_fe(e: np.ndarray, c: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Per-well-demeaned slope of ``e`` on ``c`` (wells ``w``); also the demeaned vectors."""
    n = np.bincount(w).astype(float)
    et = e - (np.bincount(w, weights=e) / n)[w]
    ct = c - (np.bincount(w, weights=c) / n)[w]
    den = float((ct ** 2).sum())
    return (float((ct * et).sum() / den) if den > 0 else float("nan")), et, ct


def boot_slope(et: np.ndarray, ct: np.ndarray, g: np.ndarray, rng) -> tuple[float, float]:
    num = np.bincount(g, weights=ct * et)
    den = np.bincount(g, weights=ct ** 2)
    ix = rng.integers(0, len(num), (N_BOOT, len(num)))
    b = num[ix].sum(1) / den[ix].sum(1)
    return float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))


def decide(arms: dict, obs: np.ndarray, test: np.ndarray, blocks: np.ndarray, rng) -> dict:
    """Scores on the (W, T) arrays of one county; ``test`` flags the scored well-months."""
    wi, _ = np.nonzero(test)
    g = blocks[wi]
    err = {k: (v - obs)[test] for k, v in arms.items()}
    out = {"n_wells": int(test.any(1).sum()), "n_well_months": int(test.sum()),
           "n_blocks": int(len(np.unique(g))),
           "rmse_m": {k: float(np.sqrt(np.mean(v ** 2))) for k, v in err.items()}, "gain": {}}
    for k in err:
        if k != "PR":
            d = err[k] ** 2 - err["PR"] ** 2
            lo, hi = boot_mean(d, g, rng)
            out["gain"][k] = {"mse_gain_m2": float(d.mean()), "lo": lo, "hi": hi}
    others = [k for k in ("Q0", "R1c", "CL") if k in err]
    out["skill"] = ("layer ahead" if all(out["rmse_m"]["PR"] < out["rmse_m"][k]
                                         and out["gain"][k]["lo"] > 0 for k in others)
                    else "layer not ahead")
    b, et, ct = slope_fe((obs - arms["ZR"])[test], (arms["PR"] - arms["ZR"])[test], wi)
    rms = float(np.sqrt(np.mean(ct ** 2)))
    lo, hi = boot_slope(et, ct, g, rng)
    out["response"] = {"slope": b, "lo": lo, "hi": hi, "rms_contrast_m": rms,
                       "informative": bool(rms >= INFORMATIVE_M)}
    zr = out["gain"]["ZR"]
    if not out["response"]["informative"]:
        v = "response not evaluable"
    elif zr["lo"] > 0 and lo > 0:
        v = "response supported, size consistent" if lo <= 1 <= hi else "response supported"
    else:
        v = "response not supported"
    out["response"]["verdict"] = v
    return out


def fit_layer(d, sel: np.ndarray, t_fit: int, swap: dict | None = None) -> dict:
    """PR, ZR and CL on the wells ``sel``; ``swap`` replaces drivers (placebo)."""
    from dataprep_staging.i1.hybrid_tf import config as C
    from dataprep_staging.i1.hybrid_tf import model as M
    T = d.y.shape[1]
    cal_d = np.asarray(d.drv_dates.month - 1)
    cal_h = cal_d[1:]
    fit_d = np.arange(len(cal_d)) <= t_fit                    # driver index = head index + 1
    dum = M.month_dummies(cal_h)
    pr, zr, cl = (np.full((len(d.sids), T), np.nan) for _ in range(3))
    lever = {}
    step = {tau: M.efilter(np.ones(240), tau) for tau in C.TAUS}
    for i in np.flatnonzero(sel):
        y = d.y[i]
        fit = np.isfinite(y) & (np.arange(T) < t_fit)
        if fit.sum() < MIN_FIT:
            continue
        ser = {n: (swap[n] if swap and n in swap else d.drivers[n][i]) for n in C.DRIVERS}
        ref = {n: float(s[fit_d].mean()) for n, s in ser.items()}
        X = np.column_stack([M.features(ser, C.DRIVERS, ref)[1:], dum])
        r = M.Ridge().fit(X[fit], y[fit])
        pr[i] = r.predict(X)
        cut = {n: (M.cut_series(s, t_fit + 1, cal_d, fit_d, False) if n in ELEC else s)
               for n, s in ser.items()}
        zr[i] = r.predict(np.column_stack([M.features(cut, C.DRIVERS, ref)[1:], dum]))
        cl[i] = M.month_clim(y, fit, cal_h)
        coef = np.zeros(X.shape[1])
        coef[r.keep] = r.beta / r.sd                          # metres per raw feature unit
        first = steady = 0.0
        for a, n in enumerate(C.DRIVERS):
            if n in ELEC:
                for b_, tau in enumerate(C.TAUS):
                    k = coef[a * len(C.TAUS) + b_] * (-CUT * ref[n])
                    steady += k
                    first += k * float(step[tau][:12].mean())
        lever[int(i)] = (first, steady)
    return {"PR": pr, "ZR": zr, "CL": cl, "lever": lever}


def load():
    from dataprep_staging.i1.hybrid_tf import config as C
    C.R0_RUN = type(C.R0_RUN)(os.path.abspath(os.path.join(RESULTS, "twin_runs",
                                                          C.R0_RUN.name)))
    from dataprep_staging.i1.hybrid_tf.data import load_design
    d = load_design(verbose=False)
    t_fit = int(np.flatnonzero(d.head_dates.strftime("%Y-%m") == FIT_END)[0]) + 1
    phys, xy = {}, np.full((len(d.sids), 2), np.nan)
    for name, run in PHYS.items():
        z = np.load(os.path.join(RESULTS, "twin_runs", run, "stage3_temporal_pred.npz"),
                    allow_pickle=True)
        if int(z["T_fit"]) != t_fit or z["pred"].shape[1] != d.y.shape[1]:
            raise SystemExit(f"{run}: holdout does not start after {FIT_END}")
        row = {str(s).zfill(8): j for j, s in enumerate(z["sids"])}
        p = np.full(d.y.shape, np.nan)
        w = np.loadtxt(os.path.join(RESULTS, "twin_runs", run, "stage3_wells.csv"),
                       delimiter=",", skiprows=1, usecols=(2, 3))
        for i, s in enumerate(d.sids):
            j = row.get(str(s).zfill(8))
            if j is not None:
                p[i] = z["pred"][j]
                xy[i] = w[j]
        phys[name] = p
    return d, t_fit, phys, xy


def units(d, t_fit: int, arms: dict, county: str) -> np.ndarray:
    T = d.y.shape[1]
    t = np.arange(T)
    ok = np.isfinite(d.y) & np.all([np.isfinite(a) for a in arms.values()], axis=0)
    test = ok & (t >= t_fit)
    keep = ((np.isfinite(d.y) & (t < t_fit)).sum(1) >= MIN_FIT) & (test.sum(1) >= MIN_SCORED)
    return test & (keep & (d.county == county))[:, None]


def run(check: bool = False) -> dict:
    rng = np.random.default_rng(SEED)
    d, t_fit, phys, xy = load()
    T = d.y.shape[1]
    fit = np.isfinite(d.y) & (np.arange(T) < t_fit)
    lay = fit_layer(d, np.ones(len(d.sids), bool), t_fit)
    arms = {k: rezero(v, fit) for k, v in {**{a: lay[a] for a in ("PR", "ZR", "CL")},
                                           **phys}.items()}
    obs = rezero(d.y, fit)
    blocks = block_ids(np.nan_to_num(xy))
    out = {"fit_end": FIT_END, "scored": f"{d.head_dates[t_fit]:%Y-%m}..{d.head_dates[-1]:%Y-%m}",
           "seed": SEED, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "counties": {}}
    for county in ("changhua", "yunlin"):
        test = units(d, t_fit, arms, county)
        if check:
            out["counties"][county] = {"n_wells": int(test.any(1).sum()),
                                       "n_well_months": int(test.sum()),
                                       "n_blocks": int(len(np.unique(blocks[test.any(1)])))}
            continue
        res = decide(arms, obs, test, blocks, rng)
        lv = np.array([lay["lever"][i] for i in np.flatnonzero(test.any(1))])
        res["lever_cut30_m_report_only"] = {
            k: {"median": float(np.median(lv[:, j])), "q25": float(np.quantile(lv[:, j], .25)),
                "q75": float(np.quantile(lv[:, j], .75)), "share_positive": float((lv[:, j] > 0).mean())}
            for j, k in enumerate(("first_year_mean", "steady_state"))}
        e = d.drivers["electricity_county"][np.flatnonzero(d.county == county)[0]]
        cal = np.asarray(d.drv_dates.month - 1)
        clim = np.array([e[: t_fit + 1][cal[: t_fit + 1] == m].mean() for m in range(12)])[cal]
        r = e / clim - 1
        res["county_electricity_anomaly_range"] = {"fit": [float(r[: t_fit + 1].min()),
                                                           float(r[: t_fit + 1].max())],
                                                   "scored": [float(r[t_fit + 1:].min()),
                                                              float(r[t_fit + 1:].max())]}
        out["counties"][county] = res
    if not check:
        ch = d.county == "changhua"
        yl_e = d.drivers["electricity_county"][np.flatnonzero(d.county == "yunlin")[0]]
        pl = fit_layer(d, ch, t_fit, swap={n: yl_e for n in ELEC})
        parms = {**arms, **{a: rezero(pl[a], fit) for a in ("PR", "ZR")}}
        p = decide(parms, obs, units(d, t_fit, arms, "changhua"), blocks, rng)
        out["placebo_yunlin_electricity_report_only"] = {
            "rmse_m": {k: p["rmse_m"][k] for k in ("PR", "ZR")}, "gain_ZR": p["gain"]["ZR"],
            "response": p["response"]}
        c = out["counties"]["changhua"]
        out["verdict"] = {"skill": c["skill"], "response": c["response"]["verdict"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="counts only; reads and prints no score")
    a = ap.parse_args()
    path = os.path.join(OUT, "result.json")
    if not a.check and os.path.exists(path):
        raise SystemExit(f"{path} exists: the retrospective test is scored once")
    res = run(check=a.check)
    if not a.check:
        os.makedirs(OUT, exist_ok=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))

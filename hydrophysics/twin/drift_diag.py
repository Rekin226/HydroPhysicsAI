"""Decompose a held-out-years head error into offset, trend, seasonal and residual parts.

The temporal gate (``calibrate_flow.temporal_gate``) says *whether* a free-running
continuation beats climatology; this module says *why not*. For each well the error
series ``e(t) = pred(t) - obs(t)`` over a window is projected, in order, onto

1. a constant (the mean level offset over the window),
2. a centred linear trend (drift),
3. an annual harmonic pair ``cos, sin(2 pi month / 12)`` (seasonal amplitude/phase),

and whatever is left is the residual. The basis is orthonormalised (Gram-Schmidt in that
order) on each well's observed months, so the four parts are exact, additive shares of
that well's mean squared error: ``offset + trend + seasonal + residual == mse``.

Everything is numpy and CPU-only; the module never touches the solver.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ("offset", "trend", "seasonal", "residual")


def _basis(t: np.ndarray, period: float) -> np.ndarray:
    """(n, 4) raw regressors: constant, centred trend, cos, sin."""
    t = np.asarray(t, dtype="float64")
    tc = t - t.mean() if t.size else t
    w = 2.0 * np.pi * t / period
    return np.stack([np.ones_like(t), tc, np.cos(w), np.sin(w)], axis=1)


def decompose_series(err: np.ndarray, t: np.ndarray | None = None,
                     period: float = 12.0) -> dict:
    """MSE shares of one error series (NaN months are skipped).

    ``t`` is the month index of each entry (calendar phase matters for the seasonal part:
    pass the absolute month so that ``t % 12`` is the calendar month). Returns the MSE,
    each component's contribution (m^2, summing to the MSE), the offset (m), the trend
    (m/yr), and the seasonal amplitude (m) of the error.
    """
    e = np.asarray(err, dtype="float64").ravel()
    t = np.arange(e.size, dtype="float64") if t is None else np.asarray(t, "float64")
    ok = np.isfinite(e)
    e, t = e[ok], t[ok]
    n = e.size
    out = {"n": int(n), "mse": float("nan"), "offset_m": float("nan"),
           "trend_m_per_yr": float("nan"), "seas_amp_m": float("nan")}
    out.update({c: float("nan") for c in COMPONENTS})
    if n < 6:
        return out
    X = _basis(t, period)
    # least-squares coefficients on the raw basis give interpretable numbers
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    out["offset_m"] = float(e.mean())
    out["trend_m_per_yr"] = float(coef[1] * 12.0)
    out["seas_amp_m"] = float(np.hypot(coef[2], coef[3]))
    # sequential orthonormal projection -> additive shares in the stated order
    Q, _ = np.linalg.qr(X)
    proj = Q.T @ e                      # coordinates on q0..q3
    shares = proj ** 2 / n
    shares = [shares[0], shares[1], shares[2] + shares[3]]
    mse = float(np.mean(e ** 2))
    out["mse"] = mse
    out["offset"], out["trend"], out["seasonal"] = (float(s) for s in shares)
    out["residual"] = float(max(mse - sum(shares), 0.0))
    return out


def decompose_wells(err: np.ndarray, t: np.ndarray | None = None,
                    period: float = 12.0) -> dict[str, np.ndarray]:
    """``decompose_series`` over every row of a ``(W, T)`` error array; returns arrays."""
    err = np.atleast_2d(np.asarray(err, dtype="float64"))
    rows = [decompose_series(err[w], t, period) for w in range(err.shape[0])]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def pooled_shares(parts: dict[str, np.ndarray],
                  mask: np.ndarray | None = None) -> dict[str, float]:
    """Month-weighted pooled MSE and the fraction of it in each component.

    Weighting each well by its number of scored months makes the pooled MSE equal the
    squared RMSE the gate reports (when no months are missing)."""
    n = parts["n"].astype("float64")
    keep = np.isfinite(parts["mse"]) & (n > 0)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if not keep.any():
        return {"n_wells": 0, "rmse": float("nan"),
                **{f"{c}_frac": float("nan") for c in COMPONENTS}}
    w = n[keep] / n[keep].sum()
    mse = float((w * parts["mse"][keep]).sum())
    res = {"n_wells": int(keep.sum()), "rmse": float(np.sqrt(mse))}
    for c in COMPONENTS:
        res[f"{c}_frac"] = float((w * parts[c][keep]).sum() / mse) if mse > 0 else 0.0
    return res


# ---------------------------------------------------------------------------------------
# Fair held-out-years verdict (round 3, 2026-09-23)
#
# The round-3 decomposition found 80-92 % of every temporal screen's held-out MSE to be a
# per-well constant offset that was already there in the fitted years, and the record
# back-fills late-starting wells with values from the held-out years. The fair verdict
# scores the continuation after removing a per-well DATUM estimated from the fitted
# months only, on observed months only, against baselines fitted on the same months.
#
# PRE-REGISTERED RULE (fixed before any screen was rescored; do not tune to results):
#     PASS  iff  rmse_datum_model <= FAIR_RMSE_K * min(baseline RMSE)
#           and  r2_shape_model   >= r2_shape_clim - FAIR_SHAPE_TOL
# with FAIR_RMSE_K = 1.25, FAIR_SHAPE_TOL = 0.05, FAIR_MIN_FIT_MONTHS = 24, and the
# baselines climatology, climatology + fitted linear trend, and persistence.
# ---------------------------------------------------------------------------------------
FAIR_RMSE_K = 1.25
FAIR_SHAPE_TOL = 0.05
FAIR_MIN_FIT_MONTHS = 24
FAIR_BASELINES = ("clim", "clim_trend", "persist")


def _month_of_year(T: int, month_offset: int) -> np.ndarray:
    return (np.arange(T) + int(month_offset)) % 12


def fair_baselines(obs: np.ndarray, T_fit: int, month_offset: int = 1
                   ) -> dict[str, np.ndarray]:
    """Per-well baselines built from the OBSERVED fitted months only -> ``(W, T)`` each.

    - ``clim``: month-of-year mean (a calendar month never observed in the fit falls back
      to the well's fitted mean);
    - ``clim_trend``: least squares of month-of-year means plus one linear trend, fitted
      jointly and extrapolated (a missing calendar month takes the mean month effect);
    - ``persist``: the last observed fitted value.

    A well with fewer than 13 observed fitted months gets ``clim_trend = clim``; a well
    with none gets NaN rows. ``month_offset``: calendar month of column ``j`` is
    ``(j + month_offset) % 12`` (the calibration target starts at the record's month 1).
    """
    obs = np.asarray(obs, dtype="float64")
    W, T = obs.shape
    moy = _month_of_year(T, month_offset)
    t = np.arange(T, dtype="float64")
    out = {k: np.full((W, T), np.nan) for k in FAIR_BASELINES}
    for w in range(W):
        o = obs[w, :T_fit]
        ok = np.isfinite(o)
        if not ok.any():
            continue
        m_fit = float(o[ok].mean())
        eff = np.full(12, np.nan)
        for mth in range(12):
            sel = ok & (moy[:T_fit] == mth)
            if sel.any():
                eff[mth] = o[sel].mean()
        clim = np.where(np.isfinite(eff), eff, m_fit)[moy]
        out["clim"][w] = clim
        out["persist"][w] = o[ok][-1]
        present = np.unique(moy[:T_fit][ok])
        if ok.sum() < 13 or present.size < 2:
            out["clim_trend"][w] = clim
            continue
        tc = t[:T_fit][ok].mean()
        X = np.column_stack([t[:T_fit][ok] - tc]
                            + [(moy[:T_fit][ok] == mth).astype("float64") for mth in present])
        coef, *_ = np.linalg.lstsq(X, o[ok], rcond=None)
        full_eff = np.full(12, float(coef[1:].mean()))
        full_eff[present] = coef[1:]
        out["clim_trend"][w] = coef[0] * (t - tc) + full_eff[moy]
    return out


def fair_temporal_verdict(pred: np.ndarray, obs: np.ndarray, T_fit: int,
                          min_fit_months: int = FAIR_MIN_FIT_MONTHS,
                          k: float = FAIR_RMSE_K, shape_tol: float = FAIR_SHAPE_TOL,
                          month_offset: int = 1) -> dict:
    """The fair held-out-years verdict on ``(W, T)`` arrays whose first ``T_fit`` months
    were fitted. ``obs`` must be the RAW record -- NaN wherever a month was never
    observed; a back-filled record defeats the point.

    1. Scoring set: wells with >= ``min_fit_months`` observed fitted months and at least
       one observed held-out month; cells: their observed held-out months. Every
       predictor is scored on exactly these cells.
    2. Datum: per well, ``d_w`` = mean of ``pred - obs`` over its observed FITTED months
       (no held-out value enters it); the model is scored as ``pred - d_w``. The level
       error (mean and RMS of ``|d_w|``) is reported separately, never hidden. Baselines
       are fitted to the same observed months, so they carry no datum (``clim`` and
       ``clim_trend`` have zero mean fitted residual by construction).
    3. Baselines: ``fair_baselines`` (climatology, climatology + fitted linear trend,
       persistence).
    4. Shape R2 = 1 - SSE / SST over the scoring cells, with SST about each well's observed
       fitted mean. On one cell set this is a monotone function of the same SSE, so the
       shape clause is an absolute (SST-scaled) tolerance against climatology, while the
       RMSE clause is a ratio against the best baseline.

    Pre-registered rule (module constants, not tuned):
    ``PASS iff rmse_datum_model <= k * best baseline RMSE and
    r2_shape_model >= r2_shape_clim - shape_tol``.
    """
    pred = np.asarray(pred, dtype="float64")
    obs = np.asarray(obs, dtype="float64")
    if pred.shape != obs.shape:
        raise ValueError(f"pred {pred.shape} and obs {obs.shape} differ")
    T_fit = int(T_fit)
    fin = np.isfinite(obs)
    n_fit = fin[:, :T_fit].sum(axis=1)
    n_held = fin[:, T_fit:].sum(axis=1)
    keep = (n_fit >= int(min_fit_months)) & (n_held > 0)
    res: dict = {"n_wells_scored": int(keep.sum()),
                 "n_wells_excluded_short_fit": int(((n_fit < min_fit_months)
                                                    & (n_held > 0)).sum()),
                 "min_fit_months": int(min_fit_months), "fair_k": float(k),
                 "fair_shape_tol": float(shape_tol)}
    if not keep.any():
        return {**res, "n_cells": 0, "verdict_fair": "NA"}
    P, Ob = pred[keep], obs[keep]
    finK = fin[keep]
    d_fit = np.where(finK[:, :T_fit], P[:, :T_fit] - Ob[:, :T_fit], 0.0)
    datum = d_fit.sum(axis=1) / finK[:, :T_fit].sum(axis=1)
    o_mean = (np.where(finK[:, :T_fit], Ob[:, :T_fit], 0.0).sum(axis=1)
              / finK[:, :T_fit].sum(axis=1))
    cells = finK[:, T_fit:]
    Oh = Ob[:, T_fit:][cells]
    sst = float(((Oh - np.broadcast_to(o_mean[:, None], cells.shape)[cells]) ** 2).sum())
    base = fair_baselines(Ob, T_fit, month_offset=month_offset)
    preds = {"model_raw": P, "model": P - datum[:, None], **base}

    def _stats(x):
        e = x[:, T_fit:][cells] - Oh
        sse = float((e ** 2).sum())
        return (float(np.sqrt(sse / e.size)), 1.0 - sse / max(sst, 1e-12),
                float(e.mean()))

    st = {name: _stats(x) for name, x in preds.items()}
    best = min(FAIR_BASELINES, key=lambda b: st[b][0])
    r_m, s_m = st["model"][0], st["model"][1]
    ok = bool(r_m <= float(k) * st[best][0] and s_m >= st["clim"][1] - float(shape_tol))
    res.update({
        "n_cells": int(cells.sum()),
        "rmse_datum_model_m": r_m, "rmse_raw_model_m": st["model_raw"][0],
        "datum_share_of_mse": (1.0 - r_m ** 2 / st["model_raw"][0] ** 2
                               if st["model_raw"][0] > 0 else 0.0),
        "level_err_mean_abs_m": float(np.abs(datum).mean()),
        "level_err_rms_m": float(np.sqrt((datum ** 2).mean())),
        "bias_datum_model_m": st["model"][2],
        "r2_shape_datum_model": s_m,
        **{f"rmse_{b}_m": st[b][0] for b in FAIR_BASELINES},
        **{f"r2_shape_{b}": st[b][1] for b in FAIR_BASELINES},
        "best_baseline": best, "rmse_best_baseline_m": st[best][0],
        "rmse_ratio_fair": r_m / st[best][0] if st[best][0] > 0 else float("inf"),
        "verdict_fair": "PASS" if ok else "FAIL"})
    return res

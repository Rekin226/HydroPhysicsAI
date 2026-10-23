"""Shared scorers (plan WP A1; sections 4.4-4.7, P4.2, P4.5).

Every work package scores with these functions, so that numbers are comparable across WPs.

Conventions
-----------
* Arrays of heads are ``(W, T)``: wells by months, NaN where unobserved. ``valid`` masks (same
  shape) restrict the cells that are scored.
* **Equal-well RMSE** (``agg="rms"``, the default) is ``sqrt(mean_w MSE_w)``: each well's
  mean squared error over its scored months, averaged over wells with equal weight, then
  rooted. This is the twin's convention (``hydrophysics.twin.hybrid._equal_well_rmse``) and the
  physics pilot's (``anom_rmse``). ``agg="mean_of_rmse"`` is the mean of per-well RMSEs, used
  by the head gap-fill pilot (``rmse_eqwell``). State which one a number uses.
* **Datum-free anomaly RMSE** removes each well's mean residual over its scored months before
  squaring (``anom_rmse`` of the pilots).
* **Out-of-time RMSE uses the training-mean datum**: each well's datum is its mean residual
  over the observed *training* months only, and the test-month errors are taken after
  subtracting it. A GLS datum (fitted jointly with the inversion, using all months) made a good
  inversion look worse than zero pumping in design pilot ``p3_results_v1_gls_datum.json``;
  it is deliberately not offered here.
* **90 % coverage** is reported both natively (``|err| <= z * sd``, ``z = 1.6448536``) and
  split-conformally (the interval scale is calibrated on one half of the exchangeable groups,
  for example gaps or sites, and checked on the other). The cluster-adjusted interval and the
  pass rule for a coverage rate are in :mod:`harness.coverage`.
* **Pseudo-truth recovery** (plan 4.5: designs are selected on recovery of pseudo-truth, never
  on fitted-period head RMSE): bias of the mean, time-variation ratio, monthly, seasonal and
  annual correlations, seasonal-amplitude ratio and block-density correlation.

Reproduction: with the pilot settings (``z=1.645``, ``finite_sample=False``, the pilot seeds)
these functions reproduce the pilot metrics to a relative 1e-6 (WP A1 acceptance; evidence in
``results/dataprep/reports/A1_evidence/``).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from common.paths import ORIGIN

#: two-sided standard-normal quantile of the central 90 % interval
Z90 = 1.6448536269514722
#: the rounded value the pilots used
Z90_PILOT = 1.645


class LeakageError(PermissionError):
    """A baseline or datum would be fitted on post-origin months (U3, U7)."""


# ----------------------------------------------------------------------------- helpers
def _2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[None, :] if a.ndim == 1 else a


def _masked_residual(obs, pred, valid=None) -> np.ndarray:
    o, p = _2d(obs), _2d(pred)
    if o.shape != p.shape:
        raise ValueError(f"obs {o.shape} and pred {p.shape} differ")
    r = o - p
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        if v.shape != r.shape:
            v = np.broadcast_to(v, r.shape)
        r = np.where(v, r, np.nan)
    return r


def _nanmean_rows(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=1)


def _aggregate(per_well_mse: np.ndarray, agg: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if agg == "rms":
            return float(np.sqrt(np.nanmean(per_well_mse)))
        if agg == "mean_of_rmse":
            return float(np.nanmean(np.sqrt(per_well_mse)))
    raise ValueError(f"agg must be 'rms' or 'mean_of_rmse', not {agg!r}")


def months_of(dates) -> np.ndarray:
    """Calendar month (1-12) of each entry of a monthly date axis."""
    return np.asarray(pd.DatetimeIndex(pd.to_datetime(dates)).month)


def month_strings(dates) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(pd.to_datetime(dates)).strftime("%Y-%m"))


# ----------------------------------------------------------------------------- head RMSE
def equal_well_rmse(obs, pred, valid=None, agg: str = "rms", min_n: int = 1) -> dict:
    """Equal-well RMSE of ``obs - pred`` over the cells marked ``valid`` (default: finite).

    Returns ``{"rmse", "n_wells", "n_cells", "per_well_mse", "per_well_n"}``. Wells with fewer
    than ``min_n`` scored months are excluded."""
    r = _masked_residual(obs, pred, valid)
    n = np.isfinite(r).sum(axis=1)
    mse = _nanmean_rows(r ** 2)
    mse = np.where(n >= max(min_n, 1), mse, np.nan)
    return {"rmse": _aggregate(mse, agg), "n_wells": int(np.isfinite(mse).sum()),
            "n_cells": int(n[np.isfinite(mse)].sum()), "per_well_mse": mse, "per_well_n": n,
            "agg": agg}


def anomaly_rmse(obs, pred, valid=None, agg: str = "rms", min_n: int = 1) -> dict:
    """Datum-free equal-well RMSE: each well's mean residual over its scored months is
    removed before squaring (the pilots' ``anom_rmse``)."""
    r = _masked_residual(obs, pred, valid)
    r = r - _nanmean_rows(r)[:, None]
    n = np.isfinite(r).sum(axis=1)
    mse = _nanmean_rows(r ** 2)
    mse = np.where(n >= max(min_n, 1), mse, np.nan)
    return {"rmse": _aggregate(mse, agg), "n_wells": int(np.isfinite(mse).sum()),
            "n_cells": int(n[np.isfinite(mse)].sum()), "per_well_mse": mse, "per_well_n": n,
            "agg": agg}


def pooled_rmse(err) -> float:
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def training_mean_datum(obs, pred, train_mask) -> tuple[np.ndarray, np.ndarray]:
    """Per-well datum = mean of ``obs - pred`` over the observed training months, and the
    number of those months (NaN datum where there are none)."""
    r = _masked_residual(obs, pred, train_mask)
    n = np.isfinite(r).sum(axis=1)
    return _nanmean_rows(r), n


def out_of_time_rmse(obs, pred, train_mask, test_mask, *, min_test: int = 7,
                     min_train: int = 1, agg: str = "rms", anomaly: bool = False) -> dict:
    """Out-of-time equal-well RMSE with the **training-mean datum**.

    ``datum_w = mean(obs - pred)`` over well w's observed training months; the score is the
    equal-well RMSE of ``obs - pred - datum_w`` over its observed test months. Wells need at
    least ``min_train`` training months and ``min_test`` test months (the pilots used
    ``> 6``, i.e. ``min_test=7``). ``anomaly=True`` also removes each well's mean test error
    (timing skill only). ``train_mask`` and ``test_mask`` are ``(T,)`` or ``(W, T)`` booleans
    and must not overlap."""
    o, p = _2d(obs), _2d(pred)
    tr = np.broadcast_to(np.asarray(train_mask, bool), o.shape)
    te = np.broadcast_to(np.asarray(test_mask, bool), o.shape)
    if np.any(tr & te):
        raise ValueError("training and test months overlap")
    datum, ntr = training_mean_datum(o, p, tr)
    e = np.where(te, (o - p) - datum[:, None], np.nan)
    nte = np.isfinite(e).sum(axis=1)
    ok = np.isfinite(datum) & (ntr >= min_train) & (nte >= min_test)
    eo = e[ok]
    if anomaly:
        eo = eo - _nanmean_rows(eo)[:, None]
    mse = _nanmean_rows(eo ** 2)
    return {"rmse": _aggregate(mse, agg), "n_wells": int(ok.sum()), "datum": datum,
            "n_train": ntr, "n_test": nte, "wells_scored": np.flatnonzero(ok),
            "anomaly": anomaly, "agg": agg}


def equal_well_rmse_long(err, wells, agg: str = "mean_of_rmse") -> float:
    """Equal-well RMSE of a long table of errors grouped by ``wells`` (default
    ``mean_of_rmse``, the gap-fill pilot's ``rmse_eqwell``; ``"rms"`` for the twin's)."""
    s = pd.Series(np.asarray(err, dtype=float)).groupby(np.asarray(wells))
    mse = s.apply(lambda x: np.mean(x ** 2)).to_numpy()
    return _aggregate(mse, agg)


def error_summary(err, thresholds=(0.25, 0.5, 1.0)) -> dict:
    """Pooled error statistics of a vector of errors (NaN-free): n, rmse, mae, median
    absolute error, bias and the share of |err| within each threshold (``le_<t>``)."""
    e = np.asarray(err, dtype=float).ravel()
    a = np.abs(e)
    out = {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(a)),
           "medae": float(np.median(a)), "bias": float(np.mean(e))}
    for t in thresholds:
        out[f"le_{str(t).replace('.', 'p')}"] = float(np.mean(a <= t))
    return out


def amplitude_ratio(obs, pred, min_n: int = 25) -> float:
    """Median over wells (with at least ``min_n`` observed months) of
    ``std(pred) / std(obs)`` on the observed months (pilot ``amp_ratio``; it used ``> 24``)."""
    o, p = _2d(obs), _2d(pred)
    out = []
    for a, b in zip(o, p, strict=True):
        m = np.isfinite(a)
        if m.sum() >= min_n:
            out.append(np.std(b[m]) / np.std(a[m]))
    return float(np.median(out)) if out else float("nan")


def climatology_baseline(obs, dates, fit_end: str = ORIGIN, *,
                         allow_post_origin: bool = False) -> np.ndarray:
    """Per-well month-of-year mean of ``obs`` over months up to ``fit_end`` (inclusive),
    applied to every month. A baseline is a pre-origin use (U3/U7): ``fit_end`` later than
    the origin raises unless ``allow_post_origin=True`` (after-the-fact analysis only)."""
    if fit_end > ORIGIN and not allow_post_origin:
        raise LeakageError(f"climatology fitted to {fit_end} > origin {ORIGIN} (U3/U7)")
    o = _2d(obs)
    ms = month_strings(dates)
    mo = months_of(dates)
    fit = ms <= fit_end
    out = np.full_like(o, np.nan)
    for m in range(1, 13):
        sel = mo == m
        if not sel.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(np.where((fit & sel)[None, :], o, np.nan), axis=1)
        out[:, sel] = mu[:, None]
    return out


def rmse_ratio(obs, pred, baseline, valid=None, agg: str = "rms") -> dict:
    """Equal-well RMSE of ``pred`` divided by that of ``baseline`` on identical cells."""
    o, p, b = _2d(obs), _2d(pred), _2d(baseline)
    v = np.isfinite(o) & np.isfinite(p) & np.isfinite(b)
    if valid is not None:
        v &= np.broadcast_to(np.asarray(valid, bool), o.shape)
    a = equal_well_rmse(o, p, v, agg)
    c = equal_well_rmse(o, b, v, agg)
    return {"ratio": a["rmse"] / c["rmse"], "rmse": a["rmse"], "rmse_baseline": c["rmse"],
            "n_wells": a["n_wells"]}


# ----------------------------------------------------------------------------- coverage
def interval_hits(err, sd, z: float = Z90) -> np.ndarray:
    """Boolean ``|err| <= z * sd`` (NaN sd counts as a miss)."""
    e = np.asarray(err, dtype=float)
    s = np.asarray(sd, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.abs(e) <= z * s


def coverage_native(err, sd, z: float = Z90, nan_sd: str = "nan") -> dict:
    """Native coverage of ``pred +- z * sd``. ``nan_sd``: ``"nan"`` returns NaN if any sd
    is missing, ``"miss"`` counts a missing sd as a miss (the pilot's arithmetic),
    ``"raise"`` raises."""
    e = np.asarray(err, dtype=float).ravel()
    s = np.asarray(sd, dtype=float).ravel()
    bad = ~np.isfinite(s)
    if bad.any():
        if nan_sd == "raise":
            raise ValueError(f"{int(bad.sum())} missing sd")
        if nan_sd == "nan":
            return {"coverage": float("nan"), "n": int(e.size), "mean_width": float("nan")}
    hits = interval_hits(e, s, z)
    return {"coverage": float(np.mean(hits)), "n": int(e.size),
            "mean_width": float(np.mean(2 * z * s)) if not bad.any() else float("nan")}


def conformal_quantile(scores, level: float = 0.90, finite_sample: bool = True) -> float:
    """Split-conformal quantile of nonconformity ``scores``. ``finite_sample=True`` takes
    the ``ceil((n+1) level)``-th smallest score (the distribution-free guarantee; ``inf``
    when that rank exceeds n); ``False`` takes ``np.quantile`` (linear), as the pilots did."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    if not finite_sample:
        return float(np.quantile(s, level))
    k = math.ceil((s.size + 1) * level)
    if k > s.size:
        return float("inf")
    return float(np.sort(s)[k - 1])


def conformal_split(err, groups, sd=None, *, level: float = 0.90, n_split: int = 20,
                    seed: int = 0, z: float = Z90, finite_sample: bool = True) -> dict:
    """Split-conformal coverage by exchangeable groups (gaps, sites, years).

    With ``sd``: the nonconformity score is ``|err / sd|``; on a random half of the groups the
    sd multiplier ``c = q_level(score) / z`` is estimated, and coverage is
    ``mean(score <= z * c)`` on the other half; averaged over ``n_split`` splits drawn with
    ``np.random.default_rng(seed)`` (groups in order of first appearance, half =
    ``max(n_groups // 2, 1)``). This is the head gap-fill pilot's ``recal`` when called with
    ``z=1.645, finite_sample=False, seed=1``. Without ``sd``: the score is ``|err|`` and the
    interval half-width is its quantile.

    Returns ``{"cov_oos", "sd_mult_full", "width_median", "n", "n_groups"}``;
    ``sd_mult_full`` is ``q_level / z`` on all groups (with ``sd``) and ``width_median`` the
    median full width ``sd * sd_mult_full * 2 * z`` (or ``2 q`` without ``sd``)."""
    e = np.asarray(err, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    nan = {"cov_oos": float("nan"), "sd_mult_full": float("nan"),
           "width_median": float("nan"), "n": int(e.size), "n_groups": 0}
    if sd is not None:
        s = np.asarray(sd, dtype=float).ravel()
        if np.isnan(s).any():
            return nan
        score = np.abs(e / s)
    else:
        s = None
        score = np.abs(e)
    gs = pd.Series(g)
    gids = pd.unique(gs)
    rng = np.random.default_rng(seed)
    qfull = conformal_quantile(score, level, finite_sample)
    cfull = qfull / z if s is not None else qfull
    cov = []
    for _ in range(n_split):
        cal = set(rng.choice(gids, max(len(gids) // 2, 1), replace=False))
        a = gs.isin(cal).to_numpy()
        q = conformal_quantile(score[a], level, finite_sample)
        if s is not None:
            c = q / z
            cov.append(np.mean(score[~a] <= z * c))
        else:
            cov.append(np.mean(score[~a] <= q))
    width = (float(np.median(s * cfull * 2 * z)) if s is not None else float(2 * cfull))
    return {"cov_oos": float(np.mean(cov)), "sd_mult_full": float(cfull),
            "width_median": width, "n": int(e.size), "n_groups": int(len(gids))}


def conformal_crossfit_hits(err, groups, sd=None, *, level: float = 0.90, n_folds: int = 5,
                            seed: int = 0, finite_sample: bool = True
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted split-conformal hits: groups are shuffled with ``default_rng(seed)`` and
    dealt into ``n_folds`` folds; each observation is scored once, with the quantile
    calibrated on the other folds. Returns ``(hits, fold_of_obs)``; feed ``hits`` and the
    groups (or coarser clusters) to :func:`harness.coverage.coverage_gate`."""
    e = np.asarray(err, dtype=float).ravel()
    g = pd.Series(np.asarray(groups).ravel())
    score = np.abs(e / np.asarray(sd, dtype=float).ravel()) if sd is not None else np.abs(e)
    gids = pd.unique(g)
    order = np.random.default_rng(seed).permutation(len(gids))
    fold_of_gid = {gids[j]: k % n_folds for k, j in enumerate(order)}
    fold = g.map(fold_of_gid).to_numpy()
    hits = np.zeros(e.size, dtype=bool)
    for f in range(n_folds):
        te = fold == f
        if not te.any():
            continue
        q = conformal_quantile(score[~te], level, finite_sample)
        hits[te] = score[te] <= q
    return hits, fold


# ----------------------------------------------------------------------------- recovery
def moy_profile(x, dates) -> np.ndarray:
    """Month-of-year mean (Jan..Dec) of a monthly series."""
    x = np.asarray(x, dtype=float)
    mo = months_of(dates)
    return np.array([x[mo == m].mean() if np.any(mo == m) else np.nan for m in range(1, 13)])


def annual_means(x, dates) -> tuple[np.ndarray, np.ndarray]:
    """(years, means) over complete calendar years only."""
    x = np.asarray(x, dtype=float)
    di = pd.DatetimeIndex(pd.to_datetime(dates))
    yrs, means = [], []
    for y in sorted(set(di.year)):
        sel = di.year == y
        if sel.sum() == 12:
            yrs.append(y)
            means.append(x[sel].mean())
    return np.array(yrs), np.array(means)


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def recovery_metrics(est, truth, dates, n_cells=None) -> dict:
    """Recovery of a pseudo-truth abstraction by an estimate (plan 4.5, P4 acceptance).

    ``est`` and ``truth`` are ``(T,)`` region totals or ``(n_blocks, T)`` block totals on the
    monthly axis ``dates``; ``n_cells`` (n_blocks,) gives block sizes for the block-density
    correlation. Matches the physics pilot's ``metrics`` (``d_pseudo_changhua.py``):
    correlations are None when the estimate is constant."""
    e, t = _2d(est), _2d(truth)
    if e.shape != t.shape:
        raise ValueError(f"est {e.shape} and truth {t.shape} differ")
    tot_e, tot_t = e.sum(0), t.sum(0)
    var = tot_e.std() > 0
    me, mt = moy_profile(tot_e, dates), moy_profile(tot_t, dates)
    _, ae = annual_means(tot_e, dates)
    _, at = annual_means(tot_t, dates)
    ratio = tot_e.mean() / tot_t.mean()
    m = {"mean_est": float(tot_e.mean()), "mean_true": float(tot_t.mean()),
         "bias": float(ratio - 1), "bias_pct": float(100 * (ratio - 1)),
         "log_error": float(np.log(ratio)) if ratio > 0 else float("-inf"),
         "monthly_corr": _corr(tot_e, tot_t) if var else None,
         "seasonal_corr": _corr(me, mt) if var else None,
         "seasonal_amp_ratio": float(np.ptp(me) / np.ptp(mt)),
         "annual_corr": _corr(ae, at) if var and ae.size > 1 else None,
         "timevar_ratio": float(np.std(tot_e - tot_e.mean()) / np.std(tot_t - tot_t.mean()))}
    if e.shape[0] > 1:
        if n_cells is None:
            raise ValueError("n_cells is needed for the block-density correlation")
        nk = np.asarray(n_cells, dtype=float)
        de, dtr = e.mean(1) / nk, t.mean(1) / nk
        m["block_density_corr"] = _corr(de, dtr) if de.std() > 0 else None
    return m


# ----------------------------------------------------------------------------- U2 weights
#: U2's rho(k): RMSE (m) of a k-clean-day partial monthly mean against the full month, from
#: the pre-origin (2012-2022) rerun of the gap-fill pilot's s4b by WP A0
#: (results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json, "preorigin").
#: The plan's values (0.33/0.49/0.64) included 2023-26 heads (U7) and are not used.
RHO_K_PREORIGIN = {3: 0.8616494466712785, 5: 0.8041236004796088, 10: 0.6505993940198698,
                   15: 0.4944701483871803, 20: 0.331143272089593}
SIGMA_OBS_FLOOR = 0.10


def sigma_obs(k, n_days):
    """U2 sigma_obs (m) of an observed monthly mean built from ``k`` clean days of an
    ``n_days`` month: ``max(0.10, rho(k))``, rho interpolated linearly in k between the
    pre-origin table points (k = 3, 5, 10, 15, 20) and from rho(20) to the 0.10 m floor at a
    complete month (k = n_days). k below 3 is outside the table and raises (such a month is
    not an observation); so does k > n_days. Broadcasts over arrays."""
    k = np.asarray(k, dtype=float)
    n = np.broadcast_to(np.asarray(n_days, dtype=float), k.shape)
    if np.any(k < 3) or np.any(k > n) or np.any(n < 28) or np.any(n > 31):
        raise ValueError("sigma_obs needs 3 <= k <= n_days and 28 <= n_days <= 31")
    xs = np.array(sorted(RHO_K_PREORIGIN), dtype=float)
    ys = np.array([RHO_K_PREORIGIN[int(x)] for x in xs])
    out = np.empty(k.shape)
    for idx in np.ndindex(k.shape):
        out[idx] = np.interp(k[idx], np.append(xs, n[idx]), np.append(ys, SIGMA_OBS_FLOOR))
    out = np.maximum(out, SIGMA_OBS_FLOOR)
    return float(out) if out.ndim == 0 else out


def u2_weight(sd, k, n_days):
    """U2 calibration weight ``sigma_obs^2 / (sigma_obs^2 + sd^2)`` of a target whose own sd
    is ``sd`` (0 for an observed month; the bridge sd for a flag-2 month)."""
    so = np.asarray(sigma_obs(k, n_days), dtype=float)
    sd = np.asarray(sd, dtype=float)
    w = so ** 2 / (so ** 2 + sd ** 2)
    return float(w) if w.ndim == 0 else w


# ----------------------------------------------------------------------------- totals
def annual_sum(df: pd.DataFrame, columns, min_count: int = 1) -> pd.DataFrame:
    """Calendar-year sums of monthly columns (index = month start), NaN when fewer than
    ``min_count`` values; adds ``months`` = number of rows in the year."""
    di = pd.DatetimeIndex(df.index)
    g = df.assign(_year=di.year).groupby("_year")
    out = g[list(columns)].sum(min_count=min_count)
    out["months"] = g.size()
    out.index.name = "year"
    return out


def ratio_of_totals(num, den):
    """Element-wise ratio of totals (series or arrays); used for coverage ratios, shares and
    the bias of a total (``ratio - 1``)."""
    return num / den

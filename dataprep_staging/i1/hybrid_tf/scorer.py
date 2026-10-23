"""Scorer and decision rule of the DRAFT prospective test (unregistered; hashed by the draft).

Test T1 (primary): one-month-ahead ARX (``prospective.predict`` ``arx1``) against the
one-month-ahead AR(1) statistical layer (``stat_ar1_1``), persistence (``persist1``) and
climatology (``clim``), all from the same frozen parameters; R0's registered forecast
(``results/twin/prospective_2026-11/predictions.csv``, a seasonal free run) is reported.

Scored well-months: window months whose head and previous-month head are both observed with
flags 0-1 (``prospective_scorer`` use), at wells with status ``fitted``. Per county:

* ``ratio_stat``  pooled RMSE(arx1) / RMSE(stat_ar1_1); also against persistence and
  climatology (and R0 where it has a prediction);
* a paired bootstrap over sites (wells of one site resampled together; 2000 draws, seed 2027)
  gives the 90 % interval of ``ratio_stat``.

A county PASSES if ``ratio_stat <= 0.97``, the bootstrap 95th percentile of ``ratio_stat`` is
below 1.00, and the ratios against persistence and climatology are below 1.00.
"""
from __future__ import annotations

import numpy as np

MIN_WELLS = {"changhua": 40, "yunlin": 50}
MIN_MONTHS_PER_WELL = 6
MIN_WINDOW_MONTHS = 9
MIN_WELLS_PER_MONTH = 100
THRESH_STAT = 0.97
N_BOOT, SEED = 2000, 2027


def decide(pred: dict, y: np.ndarray, window: np.ndarray, county: np.ndarray,
           site: np.ndarray, r0: np.ndarray | None = None) -> dict:
    """``pred`` model -> (W, T); ``y`` (W, T) observed heads (flags 0-1); ``window`` (T,)
    bool; ``county`` and ``site`` (W,). Returns per-county metrics and the outcome."""
    models = ("arx1", "stat_ar1_1", "persist1", "clim")
    ok = window[None, :] & np.isfinite(y)
    for m in models:
        ok &= np.isfinite(pred[m])
    n_w = ok.sum(axis=1)
    well_ok = n_w >= MIN_MONTHS_PER_WELL
    ok &= well_ok[:, None]
    per_month = ok.sum(axis=0)
    res = {"n_window_months_scored": int((per_month >= MIN_WELLS_PER_MONTH).sum()),
           "wells_per_month": per_month[window].astype(int).tolist(), "counties": {}}
    scorable = res["n_window_months_scored"] >= MIN_WINDOW_MONTHS
    rng = np.random.default_rng(SEED)
    passes = {}
    for c, nmin in MIN_WELLS.items():
        w = np.flatnonzero((county == c) & well_ok)
        r = {"n_wells": int(len(w)), "n_months": int(ok[w].sum())}
        if len(w) < nmin:
            scorable = False
            r["status"] = f"fewer than {nmin} wells with >= {MIN_MONTHS_PER_WELL} months"
            res["counties"][c] = r
            continue
        sse = {m: np.where(ok[w], (pred[m][w] - y[w]) ** 2, 0.0).sum(axis=1) for m in models}
        for m in models[1:]:
            r[f"ratio_{m}"] = float(np.sqrt(sse["arx1"].sum() / sse[m].sum()))
        if r0 is not None:
            o2 = ok[w] & np.isfinite(r0[w])
            e_a = np.where(o2, (pred["arx1"][w] - y[w]) ** 2, 0).sum()
            e_r = np.where(o2, (r0[w] - y[w]) ** 2, 0).sum()
            r["ratio_r0_registered"] = float(np.sqrt(e_a / e_r)) if e_r > 0 else None
        sites = np.unique(site[w])
        idx = {s: np.flatnonzero(site[w] == s) for s in sites}
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = np.concatenate([idx[s] for s in rng.choice(sites, len(sites))])
            boot[b] = np.sqrt(sse["arx1"][pick].sum() / sse["stat_ar1_1"][pick].sum())
        r["ratio_stat_ar1_1_p05"], r["ratio_stat_ar1_1_p95"] = (float(np.quantile(boot, .05)),
                                                                 float(np.quantile(boot, .95)))
        passes[c] = (r["ratio_stat_ar1_1"] <= THRESH_STAT and r["ratio_stat_ar1_1_p95"] < 1.0
                     and r["ratio_persist1"] < 1.0 and r["ratio_clim"] < 1.0)
        r["pass"] = bool(passes[c])
        res["counties"][c] = r
    if not scorable:
        res["outcome"] = "not scorable (minimum data not met; no claim either way)"
    elif all(passes.get(c, False) for c in MIN_WELLS):
        res["outcome"] = "adds skill in both counties"
    elif passes.get("changhua"):
        res["outcome"] = "adds skill in Changhua only"
    elif passes.get("yunlin"):
        res["outcome"] = "adds skill in Yunlin only"
    else:
        res["outcome"] = "no added skill"
    return res

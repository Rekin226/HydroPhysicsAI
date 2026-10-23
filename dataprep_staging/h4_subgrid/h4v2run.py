"""WP H4 repair round: the computations of the v2 tests (used by the registered run
``r2_registered.py`` and by the design-time power check ``t1_power_v2.py``).

Test units:

* **F (registered, fresh for H4):** the held-out wells: stations of heads/L1_clean/v3 that are
  not among the 158 calibration wells (the twin's theta never saw them), in an active cell,
  with a layer code that is not a proposal, and with >= 24 observed months (flags 0-1) in
  2012-02..2022-12. Multipliers are fitted on the 158 calibration wells' 2012-02..2019-12
  residuals (as v1 T1) and applied to F. Per F well, MSE over its observed 2012-02..2022-12
  months of ``obs - pred`` minus the well's own mean over those months (one free datum, the
  same for base and corrected); climatology = the well's month-of-year means over the same
  months (reported context: it cancels in the reduction).
* **F2 (reported):** the F wells with >= 24 months in 2012-02..2019-12 and >= 7 in
  2020-01..2022-12: held-out sites AND years (training-mean datum and climatology from
  2012-19, scored 2020-22).
* **v1 unit (design-time, post-look):** T1 held-out years and T2 A1 site folds at the 158
  wells, exactly as v1 (h4lib.run_t1 / run_t2) with the v2 regressors.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
import h4v2lib as V  # noqa: E402

F_WINDOW = ("2012-02", "2022-12")
F_MIN_MONTHS = 24
HIGH_MIN_WELLS = 5          # Q4 needs >= 5 scored supported F wells, else Q3+Q4 (declared)
NOHARM_MIN_WELLS = 5        # a quartile is checked on its own only with >= 5 scored F wells


def setup(cfg, with_heads: bool, cal: L.Data | None = None):
    grid = V.load_grid()
    if cal is None:
        cal = L.build_data(cfg, with_heads=with_heads)
    cal = replace(cal, cfg=cfg)
    ft, notes = V.fresh_well_table(grid, cal.sids)
    fr = V.fresh_data(cal, ft, with_heads=with_heads)
    return grid, cal, fr, notes


def scale_t_at_bound(d: L.Data, s: float) -> L.Data:
    return replace(d, T=np.where(d.t_at_bound, d.T * s, d.T)) if s != 1.0 else d


def regressors(cfg, grid, d: L.Data, tabs, **kw):
    Gs, Gl, n = V.operator_matrices(replace(d, cfg=cfg), grid, tabs, **kw)
    S, Lx = L.regressors(d, Gs, Gl)
    return [S, Lx], n


# ----------------------------------------------------------------------------- F test
def f_masks(fr: L.Data) -> dict:
    win = (fr.dates >= F_WINDOW[0]) & (fr.dates <= F_WINDOW[1])
    sc = win[None, :] & np.isfinite(fr.obs)
    n = sc.sum(1)
    return {"score": sc, "n": n, "ok": n >= F_MIN_MONTHS}


def f_mse(fr: L.Data, fm: dict, corr: np.ndarray | None) -> dict:
    pred = fr.h_cell if corr is None else fr.h_cell - corr
    r = np.where(fm["score"], fr.obs - pred, np.nan)
    moy = np.array([int(s[5:7]) for s in fr.dates])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = r - np.nanmean(r, axis=1)[:, None]
        mse = np.nanmean(e ** 2, axis=1)
        o = np.where(fm["score"], fr.obs, np.nan)
        clim = np.full(o.shape, np.nan)
        for k in range(1, 13):
            sel = moy == k
            mu = np.nanmean(o[:, sel], axis=1)
            mu = np.where(np.isfinite(mu), mu, np.nanmean(o, axis=1))
            clim[:, sel] = mu[:, None]
        msec = np.nanmean((o - clim) ** 2, axis=1)
        sd = np.nanstd(o, axis=1, ddof=1)
    ok = fm["ok"]
    return {"mse": np.where(ok, mse, np.nan), "mse_clim": np.where(ok, msec, np.nan),
            "sd": np.where(ok, sd, np.nan)}


def f_classes(sd: np.ndarray, ok: np.ndarray, support: np.ndarray, cuts) -> dict:
    """Quartile by the fixed v1 cuts on the sd over the scored months; the high-swing set is
    Q4 if it holds >= HIGH_MIN_WELLS scored supported wells, else Q3 + Q4 (declared)."""
    q = np.zeros(len(sd), int)
    q[ok] = 1 + np.searchsorted(np.asarray(cuts, float), sd[ok], side="right")
    n_q4_sup = int(((q == 4) & support).sum())
    high_q = (4,) if n_q4_sup >= HIGH_MIN_WELLS else (3, 4)
    return {"q": q, "high_q": high_q, "n_q4_supported": n_q4_sup,
            "high": ok & np.isin(q, high_q), "low_q": tuple(k for k in (1, 2, 3)
                                                             if k not in high_q)}


def f_summary(base: dict, cor: dict, cls: dict, support: np.ndarray) -> dict:
    q, ok = cls["q"], np.isfinite(base["mse"])
    out = {"high_q": list(cls["high_q"]), "n_q4_supported": cls["n_q4_supported"]}
    hi = cls["high"] & ok
    b, c, cl = base["mse"][hi], cor["mse"][hi], base["mse_clim"][hi]
    out["high"] = {"n_wells": int(hi.sum()), "n_supported": int((hi & support).sum()),
                   "rmse_base": float(np.sqrt(b.mean())) if b.size else float("nan"),
                   "rmse_corrected": float(np.sqrt(c.mean())) if c.size else float("nan"),
                   "rmse_clim": float(np.sqrt(cl.mean())) if cl.size else float("nan"),
                   "excess_base_m2": float((b - cl).sum()),
                   "excess_corrected_m2": float((c - cl).sum()),
                   "mse_reduction_m2": float((b - c).sum()),
                   "n_supported_improved": int(((base["mse"] - cor["mse"] > 0) & hi
                                                & support).sum())}
    for k in (1, 2, 3, 4):
        s = ok & (q == k)
        if not s.any():
            out[f"Q{k}"] = {"n_wells": 0}
            continue
        out[f"Q{k}"] = {"n_wells": int(s.sum()), "n_supported": int((s & support).sum()),
                        "rmse_base": float(np.sqrt(base["mse"][s].mean())),
                        "rmse_corrected": float(np.sqrt(cor["mse"][s].mean())),
                        "rmse_clim": float(np.sqrt(base["mse_clim"][s].mean())),
                        "ratio": float(np.sqrt(cor["mse"][s].mean() / base["mse"][s].mean())),
                        "mse_reduction_m2": float((base["mse"][s] - cor["mse"][s]).sum())}
    low = ok & np.isin(q, cls["low_q"])
    out["low_pooled"] = {"quartiles": list(cls["low_q"]), "n_wells": int(low.sum()),
                         "ratio": float(np.sqrt(cor["mse"][low].mean() / base["mse"][low]
                                                .mean())) if low.any() else float("nan")}
    return out


def f_noharm(summ: dict, tol: float) -> tuple[bool, str]:
    parts, ok = [], True
    lp = summ["low_pooled"]
    if lp["n_wells"]:
        ok &= lp["ratio"] <= tol
        parts.append(f"pooled Q{'+Q'.join(map(str, lp['quartiles']))} {lp['ratio']:.4f} "
                     f"(n {lp['n_wells']})")
    for k in lp["quartiles"]:
        s = summ[f"Q{k}"]
        if s.get("n_wells", 0) >= NOHARM_MIN_WELLS:
            ok &= s["ratio"] <= tol
            parts.append(f"Q{k} {s['ratio']:.4f} (n {s['n_wells']})")
        else:
            parts.append(f"Q{k} n {s.get('n_wells', 0)} < {NOHARM_MIN_WELLS}: pooled only")
    return bool(ok), "; ".join(parts)


def f2_eval(cfg, fr: L.Data, Xfr, a) -> dict:
    """Reported: F wells with fit (2012-19) and test (2020-22) months; v1-style scoring."""
    m = L.masks(replace(fr, cfg=cfg))
    clim = L.climatology(fr, m)
    corr = L.correction(Xfr, a)
    base = L.per_well_mse(fr, m, None, clim)
    cor = L.per_well_mse(fr, m, corr, clim)
    ok = m["score_ok"]
    sup = np.abs(corr).sum(1) > 0
    s = ok & sup
    return {"n_scored": int(ok.sum()), "n_scored_supported": int(s.sum()),
            "mse_reduction_supported_m2": float((base["mse"][s] - cor["mse"][s]).sum()),
            "rmse_base_supported": float(np.sqrt(np.nanmean(base["mse"][s]))) if s.any()
            else float("nan"),
            "rmse_corrected_supported": float(np.sqrt(np.nanmean(cor["mse"][s]))) if s.any()
            else float("nan"),
            "n_supported_improved": int(((base["mse"] - cor["mse"]) > 0)[s].sum()),
            "per_well": {str(fr.sids[i]): [float(base["mse"][i]), float(cor["mse"][i])]
                         for i in np.flatnonzero(s)}}


# ----------------------------------------------------------------------------- all tests
def evaluate(cfg, cal: L.Data, fr: L.Data, Xcal, Xfr, mcal, qcal, climcal, fm, cls,
             sup_fr, with_v1unit: bool = True) -> dict:
    a = L.fit_multipliers(replace(cal, cfg=cfg), Xcal, mcal, np.ones(len(cal.sids), bool))
    corr_fr = L.correction(Xfr, a)
    fb = f_mse(fr, fm, None)
    fc = f_mse(fr, fm, corr_fr)
    out = {"a": a, "corr_fr": corr_fr, "f_base": fb, "f_cor": fc,
           "f": f_summary(fb, fc, cls, sup_fr)}
    if with_v1unit:
        c2 = replace(cal, cfg=cfg)
        out["t1"] = L.run_t1(c2, Xcal, mcal, qcal, climcal)
        out["t2"] = L.run_t2(c2, Xcal, mcal, qcal, climcal)
    return out

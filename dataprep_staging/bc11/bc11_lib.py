"""Pure functions of I1.backcast_2010_11.v1 (round 8 plan #12). No function here opens a file:
the runner passes model arrays, the scorer passes the well-month arrays, the G0 studies pass
synthetic worlds, and the tests drive everything with synthetic arrays.

Axes: the fit record is 2012-01..2022-12 (132 months, column 0 = January); the window is
2010-01..2011-12 (24 months): columns 0..11 are the pre-origin year 2010 (origin 2010-12),
columns 12..23 the scored year 2011.

* G1 / G1b (primary): E04's recipe (``twin.hybrid.register``'s arithmetic) refitted on
  2012-22 only: month-of-year climatology CL (``drift_diag.fair_baselines``), lag-1 rho
  (``hybrid.lag1_rho``, pooled-median fallback), the anomaly at the last 2010 month within
  3 months of 2010-12 persisted as rho**k (``hybrid.persisted_anomaly``), the fitted-period
  residual sd (``coverage._residual_sd``) for the central 90 % interval. Scored against CL
  and the level-adjusted CL (CL + each well's mean 2010 anomaly) by the equal-station RMSE
  ratio (E02/E04's arithmetic) with a well-cluster bootstrap upper bound; G1b is the pooled
  coverage of the interval.
* G2: E07's fair rule (``drift_diag.fair_temporal_verdict``) with the baseline set
  restricted to CL: per-well datum from the fitted months only, RMSE ratio <= 1.25 and the
  shape clause, per county.
* G3: the recorded-pumping run F1 against the pumping-climatology run F0 (both datum-adjusted)
  by the equal-station ratio and its upper bound.
* Yunlin response arm: E26's primary (``chx_lib.score_backcast``, well x calendar-month and
  month-of-sample effects) on the page model's Yunlin-only contrast.
* G0: the synthetic decidability studies (FV.stat.v2's AR(1) generator for G1, G2 and G3;
  ``chx_lib.world`` for the Yunlin arm) and the shared gate rule.
* Report only: full persistence, the CF origin-update arm, the F/L memory line.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_DS = HERE.parent
for _p in (str(HERE), str(_DS / "ch_x"), str(_DS / "n2_policy"),
           str(_DS / "fv_decidability_v2"), str(_DS / "fv_decidability"), str(_DS / "e02c"),
           str(_DS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bc11_const as K  # noqa: E402
import bootstrap_scorer as BS  # noqa: E402  (pinned N2/N2C scorer, never edited)
import chx_lib as L  # noqa: E402  (E26's frozen functions, never edited)

FIT_M = L.month_list(*K.FIT)
WIN_M = L.month_list(*K.WINDOW)
RUN_M = L.month_list(*K.BC_RUN)
SCORED_M = L.month_list(*K.SCORED)
N_PRE = len(L.month_list(*K.PRE))                 # 12
if WIN_M[:N_PRE][-1] != K.ORIGIN or WIN_M[N_PRE:] != SCORED_M or len(FIT_M) != 132:
    raise ImportError("bc11 axes are inconsistent")


def moy(months) -> np.ndarray:
    """Calendar month index (0 = January) of 'YYYY-MM' strings."""
    return np.array([int(str(m)[5:7]) - 1 for m in months])


def z_level(level: int = K.LEVEL) -> float:
    from hydrophysics.twin.coverage import Z

    return float(Z[level])


# ----------------------------------------------------------------------------- E04 recipe
def fit_stat(obs_fit: np.ndarray) -> dict:
    """E04's fitted ingredients from the 2012-22 record ``(W, 132)`` (NaN = not observed):
    ``twin.hybrid.register``'s arithmetic with every 2012-22 month as the fit."""
    from hydrophysics.twin.coverage import _residual_sd
    from hydrophysics.twin.drift_diag import fair_baselines
    from hydrophysics.twin.hybrid import lag1_rho

    obs = np.asarray(obs_fit, float)
    W, T = obs.shape
    if len(FIT_M) != T:
        raise ValueError(f"the fit record has {T} months, not {len(FIT_M)}")
    clim = fair_baselines(obs, T, month_offset=0)["clim"]
    anom = obs - clim
    rho, n_pairs = lag1_rho(anom, T)
    pooled = float(np.nanmedian(rho)) if np.isfinite(rho).any() else float("nan")
    rho_used = np.where(np.isfinite(rho), rho, pooled)
    sd = _residual_sd(obs, clim, T)
    return {"cl_month": np.array(clim[:, :12]), "rho": rho, "rho_used": rho_used,
            "own_rho": np.isfinite(rho), "rho_pooled": pooled, "sd": sd,
            "n_fit": np.isfinite(obs).sum(axis=1), "n_pairs": n_pairs}


def cl_on(cl_month: np.ndarray, months) -> np.ndarray:
    """``(W, len(months))`` climatology from the per-well calendar-month values."""
    return np.asarray(cl_month, float)[:, moy(months)]


def stat_forecast(fit: dict, obs_win: np.ndarray, z: float | None = None) -> dict:
    """The E04 forecast on the window ``(W, 24)`` from the 2010 months (2011 never enters):
    CL, the persisted anomaly, the 90 % interval, the level-adjusted CL and full
    persistence. ``keep`` = hybrid.register's rule (>= 24 fitted months, finite sd, a 2010
    month within MAX_GAP of the origin)."""
    from hydrophysics.twin.hybrid import persisted_anomaly

    obs = np.asarray(obs_win, float)
    W, T = obs.shape
    if len(WIN_M) != T:
        raise ValueError(f"the window has {T} months, not {len(WIN_M)}")
    z = z_level() if z is None else z
    CL = cl_on(fit["cl_month"], WIN_M)
    anom = obs - CL
    pre = np.array(anom)
    pre[:, N_PRE:] = np.nan                          # nothing from 2011, explicitly
    a, last = persisted_anomaly(pre, N_PRE, np.asarray(fit["rho_used"], float), K.MAX_GAP)
    pred = CL + a
    sd = np.asarray(fit["sd"], float)
    with np.errstate(invalid="ignore"):
        n_pre = np.isfinite(pre[:, :N_PRE]).sum(axis=1)
        level = np.where(n_pre > 0, np.nansum(pre[:, :N_PRE], axis=1) / np.maximum(n_pre, 1),
                         np.nan)
    persist = np.full((W, T), np.nan)
    for w in np.flatnonzero(last >= 0):
        persist[w] = obs[w, last[w]]
    keep = (np.asarray(fit["n_fit"]) >= K.MIN_FIT_MONTHS) & np.isfinite(sd) & (last >= 0)
    return {"cl": CL, "pred": pred, "lower": pred - z * sd[:, None],
            "upper": pred + z * sd[:, None], "cl_adj": CL + level[:, None],
            "persist": persist, "last": last, "level": level, "keep": keep}


# ----------------------------------------------------------------------------- arithmetic
def station_msq(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-well mean squared error over the ``mask`` cells (NaN for a well with none)."""
    e2 = np.where(mask, (np.asarray(pred, float) - np.asarray(obs, float)) ** 2, 0.0)
    n = mask.sum(axis=1)
    return np.where(n > 0, e2.sum(axis=1) / np.maximum(n, 1), np.nan)


def equal_station_ratio(msq_p: np.ndarray, msq_b: np.ndarray) -> float:
    """E02/E04's equal-station RMSE ratio: sqrt(mean_w MSE_w) / sqrt(mean_w MSE_w(base))."""
    d = float(np.mean(msq_b))
    return float(math.sqrt(float(np.mean(msq_p))) / math.sqrt(d)) if d > 0 else float("inf")


def boot_index(n: int, B: int = K.BOOT_B, seed: int = K.BOOT_SEED) -> np.ndarray:
    """``(B, n)`` well-cluster resamples (wells with replacement); one draw per seed, so every
    ratio scored with the same seed uses the same resamples."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_ub(msq_p: np.ndarray, msq_b: np.ndarray, idx: np.ndarray, q: float = K.BOOT_Q
            ) -> float:
    """Upper end of the central 90 % well-cluster bootstrap interval of the ratio."""
    num = np.asarray(msq_p, float)[idx].mean(axis=1)
    den = np.asarray(msq_b, float)[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, np.sqrt(num) / np.sqrt(den), np.inf)
    return float(np.quantile(r, q))


def ratio_gate(ratio: float, ub: float, max_ratio: float = K.G1_MAX_RATIO,
               ub_below: float = K.G1_UB_BELOW) -> bool:
    """ratio <= max_ratio (ties pass) AND upper bound strictly below ub_below."""
    return bool(np.isfinite(ratio) and np.isfinite(ub) and ratio <= max_ratio
                and ub < ub_below)


def g1_gate(r_cl: float, ub_cl: float, r_adj: float, ub_adj: float) -> bool:
    """G1: the ratio gate against BOTH CL and the level-adjusted CL."""
    return ratio_gate(r_cl, ub_cl) and ratio_gate(r_adj, ub_adj)


def coverage_ok(c: float, band=K.COVERAGE_BAND) -> bool:
    """G1b: pooled coverage inside the band, bounds inclusive."""
    return bool(np.isfinite(c) and band[0] <= c <= band[1])


def lead_ratios(pred: np.ndarray, base: np.ndarray, obs: np.ndarray, mask: np.ndarray
                ) -> list:
    """Per lead month (columns), the RMSE ratio over the wells observed at that lead
    (report only: the F/L memory line)."""
    out = []
    for t in range(np.asarray(obs).shape[1]):
        m = mask[:, t]
        if not m.any():
            out.append(None)
            continue
        ep = float(np.mean((pred[m, t] - obs[m, t]) ** 2))
        eb = float(np.mean((base[m, t] - obs[m, t]) ** 2))
        out.append(math.sqrt(ep) / math.sqrt(eb) if eb > 0 else None)
    return out


# ----------------------------------------------------------------------------- G1 / G1b
def _g1_stats(fc: dict, obs_s: np.ndarray, m: np.ndarray, sel: np.ndarray, idx_seed: int,
              B: int) -> dict:
    o, mm = obs_s[sel], m[sel]
    s = slice(N_PRE, None)
    msq = {k: station_msq(fc[k][sel][:, s], o, mm) for k in ("pred", "cl", "cl_adj")}
    idx = boot_index(int(sel.sum()), B, idx_seed)
    r_cl = equal_station_ratio(msq["pred"], msq["cl"])
    r_adj = equal_station_ratio(msq["pred"], msq["cl_adj"])
    ub_cl = boot_ub(msq["pred"], msq["cl"], idx)
    ub_adj = boot_ub(msq["pred"], msq["cl_adj"], idx)
    lo, up = fc["lower"][sel][:, s], fc["upper"][sel][:, s]
    hit = (o >= lo) & (o <= up)
    cov = float(hit[mm].mean()) if mm.any() else float("nan")
    g1 = g1_gate(r_cl, ub_cl, r_adj, ub_adj)
    g1b = coverage_ok(cov)
    return {"n_wells": int(sel.sum()), "n_cells": int(mm.sum()),
            "ratio_vs_cl": r_cl, "ub90_vs_cl": ub_cl,
            "ratio_vs_cl_level_adjusted": r_adj, "ub90_vs_cl_level_adjusted": ub_adj,
            "coverage90": cov, "g1_passed": g1, "g1b_passed": g1b,
            "passed": bool(g1 and g1b)}


def g1_score(fc: dict, obs_win: np.ndarray, county, *, B: int = K.BOOT_B,
             seed: int = K.BOOT_SEED, report: bool = True) -> dict:
    """G1 + G1b on the scored year. Scored wells: ``fc['keep']`` and >= MIN_WINDOW_MONTHS
    observed 2011 months; not_evaluable below MIN_POOLED_WELLS. Pass iff G1 (both
    baselines) and G1b. County lines (same rule, county >= COUNTY_LINE_MIN_WELLS) and the
    report-only rows when ``report``."""
    obs = np.asarray(obs_win, float)
    county = np.asarray(county).astype(str)
    obs_s = obs[:, N_PRE:]
    m = np.isfinite(obs_s) & np.asarray(fc["keep"], bool)[:, None]
    use = m.sum(axis=1) >= K.MIN_WINDOW_MONTHS
    out = {"n_wells_scored": int(use.sum()),
           "dropped_well_index": [int(i) for i in np.flatnonzero(~use)],
           "min_wells": K.MIN_POOLED_WELLS}
    if use.sum() < K.MIN_POOLED_WELLS:
        return {**out, "verdict": "not_evaluable", "passed": False}
    pooled = _g1_stats(fc, obs_s, m, use, seed, B)
    out.update(pooled)
    out["verdict"] = "pass" if pooled["passed"] else "fail"
    if not report:
        return out
    lines = {}
    for c in K.COUNTIES:
        sel = use & (county == c)
        if sel.sum() < K.COUNTY_LINE_MIN_WELLS:
            lines[c] = {"n_wells": int(sel.sum()), "verdict": "not_evaluable"}
            continue
        st = _g1_stats(fc, obs_s, m, sel, seed, B)
        lines[c] = {**st, "verdict": "pass" if st["passed"] else "fail"}
    out["county_lines"] = lines
    o, mm = obs_s[use], m[use]
    s = slice(N_PRE, None)
    mp = station_msq(fc["persist"][use][:, s], o, mm)
    out["report_only"] = {
        "full_persistence_ratio_vs_cl": equal_station_ratio(
            mp, station_msq(fc["cl"][use][:, s], o, mm)),
        "full_persistence_ratio_vs_cl_level_adjusted": equal_station_ratio(
            mp, station_msq(fc["cl_adj"][use][:, s], o, mm)),
        "cl_level_adjusted_ratio_vs_cl": equal_station_ratio(
            station_msq(fc["cl_adj"][use][:, s], o, mm),
            station_msq(fc["cl"][use][:, s], o, mm)),
        "lead_ratio_stat_vs_cl": lead_ratios(fc["pred"][use][:, s], fc["cl"][use][:, s], o,
                                             mm),
        "lead_ratio_stat_vs_cl_level_adjusted": lead_ratios(
            fc["pred"][use][:, s], fc["cl_adj"][use][:, s], o, mm)}
    return out


# ----------------------------------------------------------------------------- G2
def fair_cl(model_fit: np.ndarray, obs_fit: np.ndarray, model_s: np.ndarray,
            obs_s: np.ndarray, *, k: float = K.FAIR_K, shape_tol: float = K.FAIR_SHAPE_TOL,
            min_fit: int = K.FAIR_MIN_FIT_MONTHS, shape: bool = K.G2_SHAPE_CLAUSE) -> dict:
    """E07's fair rule (``drift_diag.fair_temporal_verdict``) with the baseline set reduced
    to CL. ``model_fit``/``obs_fit`` ``(W, 132)`` 2012-22, ``model_s``/``obs_s`` ``(W, 12)``
    2011. Scoring set: >= ``min_fit`` observed fitted months and >= 1 observed 2011 month;
    datum per well = mean(model - obs) over its observed fitted months; CL = month-of-year
    mean of the observed fitted months; SST about each well's observed fitted mean.
    PASS iff rmse(model - datum) <= k * rmse(CL) [and r2_shape(model) >= r2_shape(CL) -
    shape_tol]."""
    from hydrophysics.twin.drift_diag import fair_baselines

    Pf, Of = np.asarray(model_fit, float), np.asarray(obs_fit, float)
    Ps, Os = np.asarray(model_s, float), np.asarray(obs_s, float)
    fin = np.isfinite(Of)
    keep = (fin.sum(axis=1) >= min_fit) & (np.isfinite(Os).sum(axis=1) > 0)
    res = {"n_wells_scored": int(keep.sum()), "fair_k": k, "shape_tol": shape_tol,
           "shape_clause": bool(shape), "min_fit_months": min_fit, "baselines": ["clim"]}
    if not keep.any():
        return {**res, "n_cells": 0, "passed": False, "verdict": "not_evaluable"}
    Pf, Of, Ps, Os, fin = Pf[keep], Of[keep], Ps[keep], Os[keep], fin[keep]
    nf = fin.sum(axis=1)
    datum = np.where(fin, Pf - Of, 0.0).sum(axis=1) / nf
    o_mean = np.where(fin, Of, 0.0).sum(axis=1) / nf
    clim = cl_on(fair_baselines(Of, Of.shape[1], month_offset=0)["clim"][:, :12], SCORED_M)
    cells = np.isfinite(Os)
    e_m = (Ps - datum[:, None] - Os)[cells]
    e_c = (clim - Os)[cells]
    sst = float(((Os - o_mean[:, None])[cells] ** 2).sum())
    rm = float(np.sqrt(np.mean(e_m ** 2)))
    rc = float(np.sqrt(np.mean(e_c ** 2)))
    r2m = 1.0 - float((e_m ** 2).sum()) / max(sst, 1e-12)
    r2c = 1.0 - float((e_c ** 2).sum()) / max(sst, 1e-12)
    ratio = rm / rc if rc > 0 else float("inf")
    ok = bool(np.isfinite(rm) and ratio <= k and (not shape or r2m >= r2c - shape_tol))
    return {**res, "n_cells": int(cells.sum()), "rmse_datum_model_m": rm, "rmse_clim_m": rc,
            "rmse_ratio_fair_cl": ratio, "r2_shape_datum_model": r2m, "r2_shape_clim": r2c,
            "level_err_rms_m": float(np.sqrt(np.mean(datum ** 2))), "passed": ok,
            "verdict": "pass" if ok else "fail"}


def g2_score(model_fit, obs_fit, model_s, obs_s, county) -> dict:
    """G2 per county (a county is scored only with >= COUNTY_LINE_MIN_WELLS wells in its
    scoring set; otherwise not_evaluable); the pooled line is report-only."""
    county = np.asarray(county).astype(str)
    out = {}
    for c in K.COUNTIES:
        sel = county == c
        r = fair_cl(model_fit[sel], obs_fit[sel], model_s[sel], obs_s[sel])
        if r["n_wells_scored"] < K.COUNTY_LINE_MIN_WELLS:
            r = {**r, "passed": False, "verdict": "not_evaluable"}
        out[c] = r
    out["pooled_report_only"] = fair_cl(model_fit, obs_fit, model_s, obs_s)
    return out


def physics_datum(model_fit: np.ndarray, obs_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(datum, residual sd) per well on 2012-22: datum = mean(model - obs) over observed
    months; residual sd of obs - (model - datum) (ddof 1; NaN below 12 months)."""
    Pf, Of = np.asarray(model_fit, float), np.asarray(obs_fit, float)
    fin = np.isfinite(Of)
    n = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        datum = np.where(n > 0, np.where(fin, Pf - Of, 0.0).sum(axis=1) / np.maximum(n, 1),
                         np.nan)
        r = Of - (Pf - datum[:, None])
        sd = np.nanstd(np.where(fin, r, np.nan), axis=1, ddof=1)
    sd[n < 12] = np.nan
    return datum, sd


# ----------------------------------------------------------------------------- G3
def g3_score(F1_s, F0_s, datum, obs_s, *, B: int = K.BOOT_B, seed: int = K.BOOT_SEED,
             min_wells: int = K.MIN_POOLED_WELLS) -> dict:
    """F1 (recorded pumping) against F0 (pumping climatology), both minus the 2012-22 datum,
    on the 2011 months: equal-station ratio and its well-cluster upper bound; 'value shown'
    iff ratio <= 1.0 and the bound < 1.0."""
    Os = np.asarray(obs_s, float)
    m = np.isfinite(Os) & np.isfinite(np.asarray(datum, float))[:, None]
    use = m.sum(axis=1) >= K.MIN_WINDOW_MONTHS
    if use.sum() < min_wells:
        return {"n_wells_scored": int(use.sum()), "verdict": "not_evaluable",
                "passed": False}
    d = np.asarray(datum, float)[use][:, None]
    o, mm = Os[use], m[use]
    m1 = station_msq(np.asarray(F1_s, float)[use] - d, o, mm)
    m0 = station_msq(np.asarray(F0_s, float)[use] - d, o, mm)
    r = equal_station_ratio(m1, m0)
    ub = boot_ub(m1, m0, boot_index(int(use.sum()), B, seed))
    ok = ratio_gate(r, ub, K.G3_MAX_RATIO, K.G3_UB_BELOW)
    return {"n_wells_scored": int(use.sum()), "ratio_f1_vs_f0": r, "ub90": ub,
            "passed": ok, "verdict": "value_shown" if ok else "value_not_shown"}


# ----------------------------------------------------------------------------- report only
def cf_forecast(model_fit, obs_fit, model_win, obs_win, datum) -> np.ndarray:
    """Report only, the CF origin-update arm (a STATISTICAL origin update, not the page's
    state restart): the datum-adjusted free run plus its 2010 residual at the last month
    within MAX_GAP of the origin, persisted with the residual's own 2012-22 lag-1 rho."""
    from hydrophysics.twin.hybrid import lag1_rho, persisted_anomaly

    d = np.asarray(datum, float)[:, None]
    rf = np.asarray(obs_fit, float) - (np.asarray(model_fit, float) - d)
    rho, _ = lag1_rho(rf, rf.shape[1])
    pooled = float(np.nanmedian(rho)) if np.isfinite(rho).any() else 0.0
    rho = np.where(np.isfinite(rho), rho, pooled)
    base = np.asarray(model_win, float) - d
    rw = np.asarray(obs_win, float) - base
    rw[:, N_PRE:] = np.nan
    a, _ = persisted_anomaly(rw, N_PRE, rho, K.MAX_GAP)
    return base + a


def forcing_departure_p90(F1_s, F0_s) -> list:
    """Report only (the F part of the F/L line, model only): per lead month the 90th
    percentile over the wells of |F1 - F0|."""
    return [float(np.percentile(np.abs(np.asarray(F1_s)[:, t] - np.asarray(F0_s)[:, t]), 90))
            for t in range(np.asarray(F1_s).shape[1])]


# ----------------------------------------------------------------------------- swap gate
def swap_gate_county(cn: np.ndarray, c3: np.ndarray, e_native: float, e_v3: float) -> dict:
    """The declared swap gate for one county (E26's S0 rule with this declaration's bands):
    energy ratio v3 / native in SWAP_ENERGY_BAND, per-cell-demeaned pooled correlation of
    the county-only contrasts >= SWAP_CORR_MIN, their RMS ratio in SWAP_RMS_BAND."""
    dn = cn - cn.mean(axis=-1, keepdims=True)
    d3 = c3 - c3.mean(axis=-1, keepdims=True)
    den = float(np.sqrt((dn ** 2).sum() * (d3 ** 2).sum()))
    corr = float((dn * d3).sum() / den) if den > 0 else float("nan")
    rn = float(np.sqrt(np.mean(dn ** 2)))
    rms = float(np.sqrt(np.mean(d3 ** 2)) / rn) if rn > 0 else float("nan")
    er = e_v3 / e_native if e_native > 0 else float("nan")
    ok = {"energy_ratio_ok": bool(K.SWAP_ENERGY_BAND[0] <= er <= K.SWAP_ENERGY_BAND[1]),
          "contrast_corr_ok": bool(np.isfinite(corr) and corr >= K.SWAP_CORR_MIN),
          "contrast_rms_ratio_ok": bool(K.SWAP_RMS_BAND[0] <= rms <= K.SWAP_RMS_BAND[1])}
    return {"energy_ratio_2012_22": er, "contrast_corr": corr, "contrast_rms_ratio": rms,
            **ok, "passed": all(ok.values())}


# ----------------------------------------------------------------------------- Yunlin arm
def yr_rows(P: dict, obs: np.ndarray, flag: np.ndarray | None) -> dict:
    """Long rows of the Yunlin arm (e = obs - B, c = A - B) with >= YR_MIN_MONTHS months."""
    sid = np.asarray(P["sid_yl"]).astype(str)
    months = [str(m) for m in P["months"]]
    return L.eligible(L.long_panel(sid, months, obs, P["B_yl"], P["c_yl"], flag),
                      min_months=K.YR_MIN_MONTHS)


def yr_blocks(P: dict) -> dict:
    return dict(zip(np.asarray(P["sid_yl"]).astype(str), np.asarray(P["block_yl"]).astype(str),
                    strict=True))


def score_yr(P: dict, obs: np.ndarray, flag: np.ndarray | None = None, *, B: int = BS.B_BOOT
             ) -> dict:
    """The Yunlin response arm: E26's primary on the page model's Yunlin-only contrast."""
    rows = yr_rows(P, obs, flag)
    sid = np.asarray(P["sid_yl"]).astype(str)
    scored = sorted(set(rows["well"]))
    if len(scored) < K.YR_MIN_WELLS:
        return {"verdict": "not_evaluable", "passed": False, "n_wells_scored": len(scored),
                "dropped_wells": rows["dropped_wells"]}
    keep = np.isin(sid, scored)
    blocks = yr_blocks(P)
    inf = L.informative_from_contrast(np.asarray(P["c_yl"], float)[keep], sid[keep], blocks,
                                      [str(m) for m in P["months"]], form=K.YR_FORM)
    r = L.score_backcast(rows, blocks, inf, form=K.YR_FORM, B=B)
    r["dropped_wells"] = rows["dropped_wells"]
    return r


# ----------------------------------------------------------------------------- G0
def g0_gate(power: float | None, false_pass: float | None, informative: bool = True) -> dict:
    """Decidable iff (informative and) power >= G0_POWER_MIN and false pass <=
    G0_FALSE_MAX, bounds inclusive."""
    ok = bool(informative and power is not None and false_pass is not None
              and power >= K.G0_POWER_MIN and false_pass <= K.G0_FALSE_MAX)
    return {"power": power, "false_pass": false_pass, "power_min": K.G0_POWER_MIN,
            "false_max": K.G0_FALSE_MAX, "informative": bool(informative), "decidable": ok}


def ar_world(county, rho: float, share: float, rng, T: int) -> np.ndarray:
    """FV.stat.v2's generator (stat_fv.anomalies: unit-variance AR(1), well part plus a
    county-common part with share ``share``), first ``T`` months."""
    import stat_fv as ST

    y = ST.anomalies(np.asarray(county).astype(str), rho, share, rng)
    if y.shape[1] < T:
        raise ValueError("the generator's axis is shorter than the world")
    return np.array(y[:, :T])


def g0_cells_g1() -> list[dict]:
    """G1's G0 cells, in draw order: the 9 AR cells, the 3 null cells (rho 0), then the
    report-only level-offset cells."""
    ar = [{"kind": "ar", "rho": r, "share": s, "offset_sd": 0.0}
          for r in K.G0_RHO_GRID for s in K.G0_SHARE_GRID]
    null = [{"kind": "null", "rho": K.G0_NULL_RHO, "share": s, "offset_sd": 0.0}
            for s in K.G0_SHARE_GRID]
    rep = [{"kind": "report_only_offset", "rho": 0.86, "share": 0.15, "offset_sd": o}
           for o in K.G0_G1_OFFSET_SDS]
    return ar + null + rep


def g1_world_score(y: np.ndarray, mask_win: np.ndarray, county, boot_seed: int,
                   B: int = K.BOOT_B) -> dict:
    """One synthetic world ``(W, 156)``: columns 0..23 = 2010-11 (masked by ``mask_win``),
    24..155 = 2012-22 (all observed; optimistic) -> G1 / G1b exactly as scored."""
    obs_win = np.where(mask_win, y[:, :len(WIN_M)], np.nan)
    fc = stat_forecast(fit_stat(y[:, len(WIN_M):]), obs_win)
    return g1_score(fc, obs_win, county, B=B, seed=boot_seed, report=False)


def cell_gate(rows: list[dict], key: str, power_kind: str, null_kind: str) -> dict:
    """The G0 gate from per-cell rates: power = equal-weight MEAN of ``key`` over the cells
    of kind ``power_kind`` only, false pass = MAX over the cells of kind ``null_kind`` only;
    report-only cells (any other kind) never enter either."""
    pw = [float(r[key]) for r in rows if r["kind"] == power_kind]
    nl = [float(r[key]) for r in rows if r["kind"] == null_kind]
    if not pw or not nl:
        raise ValueError("a G0 gate needs power and null cells")
    return g0_gate(float(np.mean(pw)), float(max(nl)))


def run_g0_g1(county, mask_win: np.ndarray, *, reps: int = K.G0_K_G1,
              seed: int = K.G0_SEEDS["g1"], B: int = K.BOOT_B) -> dict:
    """G1's G0: P(primary pass) per cell; power = equal-weight mean over the 9 AR cells,
    false pass = max over the 3 null cells (``cell_gate``); generators
    SeedSequence(seed).spawn(n cells) in ``g0_cells_g1`` order, each world's bootstrap seed
    drawn from its cell's generator. Worlds are optimistic: 2012-22 fully observed and the
    2010-11 months are the availability look's (an upper bound on the QC'd months)."""
    county = np.asarray(county).astype(str)
    T = len(WIN_M) + len(FIT_M)
    cells = g0_cells_g1()
    ss = np.random.SeedSequence(seed).spawn(len(cells))
    out = []
    for c, s in zip(cells, ss, strict=True):
        rng = np.random.default_rng(s)
        n = {"passed": 0, "g1": 0, "g1b": 0, "g1_cl_only": 0, "not_evaluable": 0}
        r_cl, r_adj, cov = [], [], []
        for _ in range(reps):
            y = ar_world(county, c["rho"], c["share"], rng, T)
            if c["offset_sd"]:
                y[:, :len(WIN_M)] += c["offset_sd"] * rng.standard_normal(len(county))[:, None]
            r = g1_world_score(y, mask_win, county, int(rng.integers(2 ** 31 - 1)), B)
            if r["verdict"] == "not_evaluable":
                n["not_evaluable"] += 1
                continue
            n["passed"] += r["passed"]
            n["g1"] += r["g1_passed"]
            n["g1b"] += r["g1b_passed"]
            n["g1_cl_only"] += ratio_gate(r["ratio_vs_cl"], r["ub90_vs_cl"])
            r_cl.append(r["ratio_vs_cl"])
            r_adj.append(r["ratio_vs_cl_level_adjusted"])
            cov.append(r["coverage90"])
        out.append({**c, "replicates": reps, "p_pass": n["passed"] / reps,
                    "p_g1": n["g1"] / reps, "p_g1b": n["g1b"] / reps,
                    "p_g1_vs_cl_only": n["g1_cl_only"] / reps,
                    "n_not_evaluable": n["not_evaluable"],
                    "median_ratio_vs_cl": _median(r_cl),
                    "median_ratio_vs_cl_level_adjusted": _median(r_adj),
                    "median_coverage90": _median(cov)})
    return {"cells": out, "gate": cell_gate(out, "p_pass", "ar", "null"), "reps": reps,
            "seed": seed, "boot_B": B,
            "power_rule": "equal-weight mean of P(G1 and G1b) over the 9 AR cells",
            "false_rule": "max of P(G1 and G1b) over the 3 rho = 0 cells",
            "optimistic": "2012-22 fully observed; 2010-11 months from the availability look"}


def _median(v) -> float | None:
    return float(np.median(v)) if len(v) else None


def phys_world(model_fit, model_s, sigma, county, rho, share, rng, truth_s) -> tuple:
    """One synthetic world for G2 / G3: AR noise (FV.stat.v2's generator x per-well
    ``sigma``) drawn in TIME ORDER, 2011 first (its first 12 months) then 2012-22, on
    ``truth_s`` in 2011 (the world's true 2011 heads) and on the model's 2012-22 heads (the
    model fits its fitted years)."""
    nf = np.asarray(model_fit).shape[1]
    ns = len(SCORED_M)
    eps = ar_world(county, rho, share, rng, ns + nf) * np.asarray(sigma)[:, None]
    return np.asarray(model_fit, float) + eps[:, ns:], np.asarray(truth_s, float) + eps[:, :ns]


def model_clim_s(model_fit) -> np.ndarray:
    """The model's own 2012-22 month-of-year mean on the 2011 months (the null world's
    truth: climatology right, the model's 2011 anomaly carries no information)."""
    mf = np.asarray(model_fit, float)
    return cl_on(np.stack([mf[:, moy(FIT_M) == k].mean(axis=1) for k in range(12)], axis=1),
                 SCORED_M)


def discontinuity(F1_last, native_first) -> dict:
    """Report only, model only: F1 at 2011-12 minus the page forward at 2012-01 at each
    screen (the spin-up / placement discontinuity the 2012-22 datum cannot see)."""
    d = np.asarray(F1_last, float) - np.asarray(native_first, float)
    return {"n": int(d.size), "mean_m": float(np.mean(d)), "median_m": float(np.median(d)),
            "rms_m": float(np.sqrt(np.mean(d ** 2))),
            "p90_abs_m": float(np.percentile(np.abs(d), 90)),
            "max_abs_m": float(np.max(np.abs(d)))}


def g2_g0_cells() -> list[tuple]:
    """(kind, rho, share): 9 power cells, 9 null cells, then the report-only power cell
    with the datum discontinuity."""
    grid = [(r, s) for r in K.G0_RHO_GRID for s in K.G0_SHARE_GRID]
    return ([("power", r, s) for r, s in grid] + [("null", r, s) for r, s in grid]
            + [("report_only_discontinuity", 0.86, 0.15)])


def run_g0_g2(model_fit, model_s, sigma, county, mask_s, *, reps: int = K.G0_K_G2,
              seed: int = K.G0_SEEDS["g2"], disc=None) -> dict:
    """G2's G0 per county: power = mean over the 9 power cells of P(G2 pass | 2011 truth =
    the model + noise) -- about 1 by construction, so decidability rests on the false pass
    --, false pass = max over the 9 null cells of P(G2 pass | 2011 truth = the model's own
    2012-22 climatology + noise; it falls as F1's 2011 departure from that climatology,
    spin-up drift included, grows). Report only: a power cell whose 2011 truth is the model
    minus its 2011-12 -> 2012-01 discontinuity ``disc`` (truth continuous with the native
    run; the datum from 2012-22 cannot remove it)."""
    county = np.asarray(county).astype(str)
    cells = g2_g0_cells()
    ss = np.random.SeedSequence(seed).spawn(len(cells))
    ms = np.asarray(model_s, float)
    truths = {"power": ms, "null": model_clim_s(model_fit),
              "report_only_discontinuity": ms - (0.0 if disc is None else
                                                 np.asarray(disc, float)[:, None])}
    rows = []
    for (kind, r, s), sq in zip(cells, ss, strict=True):
        rng = np.random.default_rng(sq)
        n = dict.fromkeys(K.COUNTIES, 0)
        for _ in range(reps):
            of, os_ = phys_world(model_fit, model_s, sigma, county, r, s, rng, truths[kind])
            res = g2_score(np.asarray(model_fit, float), of, ms,
                           np.where(mask_s, os_, np.nan), county)
            for c in K.COUNTIES:
                n[c] += bool(res[c]["passed"])
        rows.append({"kind": kind, "rho": r, "share": s, "replicates": reps,
                     **{f"p_pass_{c}": n[c] / reps for c in K.COUNTIES}})
    gate = {c: cell_gate(rows, f"p_pass_{c}", "power", "null") for c in K.COUNTIES}
    return {"cells": rows, "gate": gate, "reps": reps, "seed": seed,
            "discontinuity_supplied": disc is not None}


def run_g0_g3(model_fit, F1_s, F0_s, sigma, county, mask_s, *, reps: int = K.G0_K_G3,
              seed: int = K.G0_SEEDS["g3"], B: int = K.BOOT_B) -> dict:
    """G3's G0 (pooled): power = mean over the 9 power cells of P(value shown | 2011 truth
    = F1 + noise), false pass = max over the 9 null cells of P(value shown | truth = F0 +
    noise)."""
    county = np.asarray(county).astype(str)
    cells = [(kind, r, s) for kind in ("power", "null") for r in K.G0_RHO_GRID
             for s in K.G0_SHARE_GRID]
    ss = np.random.SeedSequence(seed).spawn(len(cells))
    mf = np.asarray(model_fit, float)
    truths = {"power": np.asarray(F1_s, float), "null": np.asarray(F0_s, float)}
    rows = []
    for (kind, r, s), sq in zip(cells, ss, strict=True):
        rng = np.random.default_rng(sq)
        n = 0
        for _ in range(reps):
            of, os_ = phys_world(mf, F1_s, sigma, county, r, s, rng, truths[kind])
            datum, _ = physics_datum(mf, of)
            res = g3_score(F1_s, F0_s, datum, np.where(mask_s, os_, np.nan), B=B,
                           seed=int(rng.integers(2 ** 31 - 1)))
            n += bool(res["passed"])
        rows.append({"kind": kind, "rho": r, "share": s, "replicates": reps,
                     "p_pass": n / reps})
    return {"cells": rows, "gate": cell_gate(rows, "p_pass", "power", "null"), "reps": reps,
            "seed": seed, "boot_B": B}


def yr_g0_panel(P: dict, mask: np.ndarray) -> dict:
    """The arm's panel in chx_lib.world's layout (confounder correlated with the Yunlin
    county pumping anomaly; no common trend)."""
    c = np.asarray(P["c_yl"], float)
    months = [str(m) for m in P["months"]]
    return {"c": c, "mask": np.asarray(mask, bool), "a_ch": np.asarray(P["a_yl"], float),
            "block": np.asarray(P["block_yl"]).astype(str),
            "sid": np.asarray(P["sid_yl"]).astype(str), "months": months,
            "hinge": np.zeros(len(months)), "trend_sign": 1.0, "load_z": L.load_z(c)}


def yr_plumbing(pan: dict) -> dict:
    """No noise: the arm's slope must equal k exactly (k = 0.5, 1, 2)."""
    out = {}
    for k in (0.5, 1.0, 2.0):
        e = np.where(pan["mask"], k * pan["c"], np.nan)
        r = L.eligible(L._panel_rows(pan, e), min_months=K.YR_MIN_MONTHS)
        et, ct = L.transform(r["obs"] - r["zr"], r["c"], r, K.YR_FORM)
        out[str(k)] = BS.slope_fe(et, ct, r["well"])
    ok = all(np.isfinite(v) and abs(v - float(k)) <= 1e-9 for k, v in out.items())
    return {"betas": out, "ok": bool(ok)}


def run_g0_yr(P: dict, mask: np.ndarray, params: dict, *, reps: int = K.G0_K_YR,
              seed: int = K.G0_SEEDS["yr"], B: int = BS.B_BOOT) -> dict:
    """The Yunlin arm's own G0 (R8.6c): informative gate on the transformed contrast first
    (if it fails: not decidable, nothing run), exact plumbing, then the cells k in G0_YR_K x
    rho in G0_YR_RHO (k outer) and the report-only noise cell; power = P(pass | k 1, rho
    -0.83), false pass = max over rho of P(pass | k 0)."""
    pan = yr_g0_panel(P, mask)
    blocks = dict(zip(pan["sid"], pan["block"], strict=True))
    inf = L.informative_from_contrast(pan["c"], pan["sid"], blocks, pan["months"],
                                      form=K.YR_FORM)
    if not inf["passes"]:
        return {"informative": inf, "cells": [], "gate": g0_gate(None, None, False)}
    plumb = yr_plumbing(pan)
    if not plumb["ok"]:
        return {"informative": inf, "plumbing": plumb, "cells": [], "engineering_failure": True,
                "gate": g0_gate(None, None, True)}
    grid = [(k, r, 1.0) for k in K.G0_YR_K for r in K.G0_YR_RHO]
    grid.append((1.0, -0.83, K.G0_YR_REPORT_NOISE))
    ss = np.random.SeedSequence(seed).spawn(len(grid))
    cells = []
    for (k, r, ns), sq in zip(grid, ss, strict=True):
        rng = np.random.default_rng(sq)
        n = 0
        for _ in range(reps):
            e = L.world(pan, k, r, rng, params, trend_mult=0.0, noise_scale=ns)
            rows = L.eligible(L._panel_rows(pan, e), min_months=K.YR_MIN_MONTHS)
            if len(set(rows["well"])) < K.YR_MIN_WELLS:
                continue
            n += L.score_backcast(rows, blocks, inf, form=K.YR_FORM, B=B,
                                  report=False)["passed"]
        cells.append({"k": k, "rho": r, "noise_scale": ns, "replicates": reps,
                      "report_only": ns != 1.0, "p_pass": n / reps})

    return {"informative": inf, "plumbing": plumb, "cells": cells, "gate": yr_gate(cells),
            "reps": reps, "seed": seed, "boot_B": B}


def yr_gate(cells: list[dict]) -> dict:
    """The arm's G0 gate: power = P(pass | k 1, rho -0.83, declared noise), false pass = max
    over rho in G0_YR_RHO of P(pass | k 0, declared noise); report-only cells never enter."""
    def p(k, r):
        return next(c["p_pass"] for c in cells if not c["report_only"]
                    and c["k"] == k and c["rho"] == r)
    return g0_gate(p(1.0, -0.83), max(p(0.0, r) for r in K.G0_YR_RHO))

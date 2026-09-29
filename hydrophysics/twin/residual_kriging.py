"""Model + residual interpolation (regression kriging) at unseen wells (2026-09-29).

    python -m hydrophysics.twin.residual_kriging results/twin_runs/stage3_datum_gate \\
        --out results/twin_runs/stage3_datum_gate_rk

**The question.** At wells a k-fold hides, the per-well datum model's head-change
(own-mean anomaly) R2 is +0.34 against +0.61 for inverse-distance interpolation of the
neighbours' heads (``kfold_scores``). The model carries the pumping -> head physics; IDW
carries the local timing the neighbours record. The classical way to combine them is
regression kriging: predict with the model, then interpolate the model's *residuals* at
the observed wells and add them back.

**What this module does, per fold of the gate's own partition** (read from
``stage3_per_entry.npz``, so the held-out sets are exactly the ones the gate scored):

1. rebuild the fold's parameter set (``stage3_fold_thetas.json``) with
   ``forward.build_model`` and hindcast it on CPU from the fold's KEPT-wells initial field
   (``TwinInputs.initial_heads(0, well_mask=keep)``, as ``calibrate_flow.kfold_wells``
   builds it) -> model heads at every well. The rebuilt held-out predictions are checked
   against the dump (``max_abs_diff_vs_dump_m``);
2. at the fold's kept wells form residual anomalies ``r = (obs - mean obs) - (model -
   mean model)``, each over that well's own finite months. The datum is a constant per
   well, so it drops out of every anomaly and never enters here;
3. interpolate ``r`` to the held-out wells, month by month, from kept wells only: IDW with
   the gate's weights (``subsidence.idw_interp``, power 2, every layer pooled, at the
   wells' cell centroids), and optionally simple kriging with an exponential covariance
   fitted to the kept wells' residuals of that fold only;
4. score ``model + correction`` with ``kfold_scores.kfold_anomaly_scores`` (same months,
   same own-mean de-meaning, same IDW reference) beside the model alone and the IDW of the
   observed anomalies.

Model-free controls use the same interpolators on the observed anomalies alone
(``idw_anom_layer``, ``sk_anom``); a hybrid's margin over its control is what the model
itself contributes (``rk_bootstrap_vs_control.csv``).

A held-out well never informs its own correction, its fold's covariance fit, or its
fold's initial field.

**Result on ``stage3_datum_gate`` (2026-09-29).** Anomaly R2 at held-out wells: model
+0.342, gate IDW +0.605, regression kriging +0.599 (IDW) / +0.622 (per-layer IDW) /
+0.635 (simple kriging); but the model-free controls score +0.654 (per-layer IDW) and
+0.643 (simple kriging), so every hybrid is at or below its own control (-0.006, -0.033,
-0.008; bootstrap over locations). The model's local departure from its own
interpolation correlates 0.06-0.28 with the observed one: it carries no usable head-change
information between wells beyond the neighbours'. Its error there is mostly amplitude
(58 % of the anomaly SSE; the model's swings are 0.62x the observed, 0.25x in the
proximal fan), then phase (30 %), then trend (12 %). ``diagnose`` splits the model's anomaly error at held-out wells into
trend, amplitude and phase by zone and layer.

``correct_head_field`` applies the same correction to a head field ``(L, A, T)`` (every
cell gets the IDW of the observed wells' residual anomalies of its month); it is the
opt-in display correction ``python -m hydrophysics.twin.residual_kriging correct`` writes
beside a finished forward run. It is never applied to the heads that drive the
compaction column.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ..subsidence import idw_interp
from .kfold_scores import _per_row_r2, _r2, kfold_anomaly_scores, own_mean_anomaly, scored_mask

IDW_POWER = 2.0            # subsidence.idw_interp's default, the gate's IDW baseline
PER_ENTRY = "stage3_per_entry.npz"
# the model-free controls (idw_anom_layer, sk_anom) use the same interpolator as the
# hybrid beside them, so a hybrid's gain over its control is the model's contribution
METHODS = ("model", "idw_heads", "idw_anom", "idw_anom_layer", "sk_anom", "rk_idw",
           "rk_idw_layer", "rk_sk", "rk_beta")


# ---------------------------------------------------------------------------------------
# residual anomalies and their interpolation (numpy only)
# ---------------------------------------------------------------------------------------
def row_anomaly(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Each row minus its own mean over ``mask`` (default: its finite months); NaN
    elsewhere. The same rule as ``kfold_scores.own_mean_anomaly``."""
    x = np.asarray(x, dtype="float64")
    if mask is None:
        mask = np.isfinite(x)
    return own_mean_anomaly(x, mask & np.isfinite(x))


def residual_anomalies(model: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """``(S, T)`` residual anomaly ``(obs - mean obs) - (model - mean model)`` of each
    source well, both means over the months where BOTH are finite; NaN elsewhere. A
    per-well constant (the datum, a level bias) cancels exactly."""
    model, obs = np.asarray(model, dtype="float64"), np.asarray(obs, dtype="float64")
    m = np.isfinite(model) & np.isfinite(obs)
    return row_anomaly(obs, m) - row_anomaly(model, m)


def idw_correction(tgt_xy: np.ndarray, src_xy: np.ndarray, resid: np.ndarray,
                   power: float = IDW_POWER, src_layer: np.ndarray | None = None,
                   tgt_layer: np.ndarray | None = None) -> np.ndarray:
    """IDW of the sources' residual anomalies ``(S, T)`` to the targets -> ``(N, T)``.
    With ``src_layer``/``tgt_layer`` each target draws only on sources of its own layer
    (falling back to every source when its layer has none)."""
    if src_layer is None or tgt_layer is None:
        return idw_interp(tgt_xy, src_xy, resid, power=power)
    out = np.full((len(tgt_xy), resid.shape[1]), np.nan)
    allsrc = idw_interp(tgt_xy, src_xy, resid, power=power)
    for k in np.unique(tgt_layer):
        t = tgt_layer == k
        s = src_layer == k
        out[t] = (idw_interp(tgt_xy[t], src_xy[s], resid[s], power=power) if s.any()
                  else allsrc[t])
    return out


def empirical_semivariogram(xy: np.ndarray, resid: np.ndarray, n_bins: int = 15,
                            max_lag_m: float | None = None) -> pd.DataFrame:
    """Pooled-over-months semivariogram of residual anomalies: per well pair,
    ``gamma = 0.5 * mean_t (r_i - r_j)^2`` over months both have; binned by distance.
    Pairs at zero distance (co-located screens) are the nugget bin."""
    xy = np.asarray(xy, dtype="float64")
    S = len(xy)
    iu, ju = np.triu_indices(S, k=1)
    d = np.sqrt(((xy[iu] - xy[ju]) ** 2).sum(-1))
    diff2 = (resid[iu] - resid[ju]) ** 2
    n = np.isfinite(diff2).sum(1)
    g = 0.5 * np.divide(np.nansum(diff2, 1), n, out=np.full(len(n), np.nan), where=n > 0)
    ok = np.isfinite(g)
    d, g, n = d[ok], g[ok], n[ok]
    if max_lag_m is None:
        max_lag_m = 0.5 * float(d.max()) if d.size else 1.0
    edges = np.concatenate([[0.0, 1.0], np.linspace(1.0, max_lag_m, n_bins)[1:]])
    rows = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = (d >= a) & (d < b) if a > 0 else (d < b)
        if sel.any():
            rows.append({"lag_m": float(d[sel].mean()), "gamma": float(np.average(
                g[sel], weights=n[sel])), "n_pairs": int(sel.sum())})
    return pd.DataFrame(rows)


def fit_exponential(vg: pd.DataFrame) -> dict:
    """Weighted least squares ``gamma(h) = nugget + psill * (1 - exp(-h / range))`` on a
    small grid of ranges (closed form in nugget/psill for each), pair counts as weights."""
    h, g, w = (vg[c].to_numpy(dtype="float64") for c in ("lag_m", "gamma", "n_pairs"))
    best = None
    for rng in np.geomspace(500.0, 60000.0, 120):
        X = np.stack([np.ones_like(h), 1.0 - np.exp(-h / rng)], 1)
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], g * sw, rcond=None)
        nug, ps = max(float(coef[0]), 0.0), max(float(coef[1]), 1e-9)
        sse = float((w * (nug + ps * (1.0 - np.exp(-h / rng)) - g) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"nugget": nug, "psill": ps, "range_m": float(rng), "sse": sse}
    return best


def _exp_cov(d: np.ndarray, vp: dict) -> np.ndarray:
    """Source-source covariance ``psill exp(-d / range)`` plus the nugget on the diagonal
    ONLY. Two distinct wells in one cell (nested screens share a cell centroid) are the
    variogram's first bin, whose gamma is the nugget: their covariance is ``psill``, not
    ``psill + nugget`` (which would make the pair perfectly correlated and ``C`` singular)."""
    c = vp["psill"] * np.exp(-d / vp["range_m"])
    if c.ndim == 2 and c.shape[0] == c.shape[1]:
        c = c + vp["nugget"] * np.eye(c.shape[0])
    return c


def simple_kriging(tgt_xy: np.ndarray, src_xy: np.ndarray, resid: np.ndarray,
                   vp: dict) -> np.ndarray:
    """Simple kriging (known mean 0: the residuals are anomalies) of ``resid`` ``(S, T)``
    to the targets under the exponential covariance ``vp``, month by month over each
    month's finite sources (one solve per distinct availability pattern)."""
    src_xy, tgt_xy = np.asarray(src_xy, "float64"), np.asarray(tgt_xy, "float64")
    dss = np.sqrt(((src_xy[:, None] - src_xy[None]) ** 2).sum(-1))
    dts = np.sqrt(((tgt_xy[:, None] - src_xy[None]) ** 2).sum(-1))
    C, c0 = _exp_cov(dss, vp), vp["psill"] * np.exp(-dts / vp["range_m"])
    fin = np.isfinite(resid)
    out = np.full((len(tgt_xy), resid.shape[1]), np.nan)
    patterns: dict[bytes, list[int]] = {}
    for t in range(resid.shape[1]):
        patterns.setdefault(fin[:, t].tobytes(), []).append(t)
    for key, ts in patterns.items():
        s = np.frombuffer(key, dtype=bool)
        if not s.any():
            continue
        Cs = C[np.ix_(s, s)] + 1e-9 * np.eye(int(s.sum()))
        lam = np.linalg.solve(Cs, c0[:, s].T).T                       # (N, S_t)
        out[:, ts] = lam @ resid[np.ix_(s, ts)]
    return out


# ---------------------------------------------------------------------------------------
# diagnosis of the model's anomaly error (part a)
# ---------------------------------------------------------------------------------------
def _trend(a: np.ndarray) -> np.ndarray:
    """Least-squares straight line through each row's finite months (NaN elsewhere)."""
    out = np.full_like(a, np.nan)
    t = np.arange(a.shape[1], dtype="float64")
    for i in range(a.shape[0]):
        m = np.isfinite(a[i])
        if m.sum() >= 2:
            out[i, m] = np.polyval(np.polyfit(t[m], a[i, m], 1), t[m])
    return out


def _slope_m_yr(line: np.ndarray, idx: np.ndarray) -> float:
    """Slope (m/yr) of a monthly straight line sampled at ``idx``."""
    return float((line[idx[-1]] - line[idx[0]]) / max(int(idx[-1] - idx[0]), 1) * 12.0)


def error_decomposition(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Per row: the anomaly MSE of ``pred`` against ``obs`` over ``mask`` split into
    ``trend`` (the difference of the two straight-line fits) plus the detrended part,
    itself split exactly into ``amplitude`` ``(sd_p - sd_o)^2`` and ``phase``
    ``2 sd_p sd_o (1 - corr)`` of the detrended series. Also the sd of both anomalies,
    their ratio, the correlation and both trends (m/yr)."""
    ap, ao = own_mean_anomaly(pred, mask), own_mean_anomaly(obs, mask)
    tp, to = _trend(ap), _trend(ao)
    dp, do = ap - tp, ao - to
    rows = []
    for i in range(ap.shape[0]):
        m = np.isfinite(ap[i]) & np.isfinite(ao[i])
        if m.sum() < 3:
            rows.append({})
            continue
        sp, so = float(dp[i, m].std()), float(do[i, m].std())
        corr = (float(np.corrcoef(dp[i, m], do[i, m])[0, 1])
                if sp > 1e-12 and so > 1e-12 else 0.0)
        mse = float(((ap[i, m] - ao[i, m]) ** 2).mean())
        tr = float(((tp[i, m] - to[i, m]) ** 2).mean())
        idx = np.flatnonzero(m)
        rows.append({"mse_anom": mse, "mse_trend": tr,
                     "mse_amplitude": (sp - so) ** 2, "mse_phase": 2 * sp * so * (1 - corr),
                     "sd_anom_model": float(ap[i, m].std()), "sd_anom_obs": float(
                         ao[i, m].std()), "sd_ratio": float(ap[i, m].std()) / max(
                         float(ao[i, m].std()), 1e-9),
                     "corr_detrended": corr, "trend_model_m_yr": _slope_m_yr(tp[i], idx),
                     "trend_obs_m_yr": _slope_m_yr(to[i], idx)})
    return pd.DataFrame(rows)


def diagnose(per_entry: dict, zone_of_entry: np.ndarray, zone_names: tuple[str, ...]
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per held-out well: the model's and IDW's anomaly error decomposition, with zone and
    layer; and its pooled summary by (zone, layer). Pooled R2 per group uses the group's
    own obs variance (so a group R2 is 'how much of that group's head-change variance')."""
    pred, idw, obs = per_entry["pred"], per_entry["idw"], per_entry["obs"]
    mask = scored_mask(pred, idw, obs)
    dm = error_decomposition(pred, obs, mask).add_suffix("_model")
    di = error_decomposition(idw, obs, mask)[["mse_anom", "sd_ratio", "corr_detrended"]
                                            ].add_suffix("_idw")
    ao = own_mean_anomaly(obs, mask)
    wells = pd.DataFrame({"entry": per_entry["entry"], "fold": per_entry["fold"],
                          "zone": [zone_names[z] for z in zone_of_entry],
                          "layer": per_entry["layer"] + 1,
                          "r2_anom_model": _per_row_r2(own_mean_anomaly(pred, mask), ao),
                          "r2_anom_idw": _per_row_r2(own_mean_anomaly(idw, mask), ao)})
    wells = pd.concat([wells, dm, di], axis=1)
    wells["ss_obs"] = np.nansum(ao ** 2, axis=1)
    wells["n_months"] = mask.sum(1)
    g = []
    for (z, k), d in list(wells.groupby(["zone", "layer"])) + [(("all", 0), wells)]:
        n = d["n_months"].to_numpy()
        tot = float((d["mse_anom_model"] * n).sum())
        g.append({"zone": z, "layer": k, "n_wells": len(d),
                  "r2_anom_model": 1 - tot / max(float(d["ss_obs"].sum()), 1e-12),
                  "r2_anom_idw": 1 - float((d["mse_anom_idw"] * n).sum())
                  / max(float(d["ss_obs"].sum()), 1e-12),
                  "share_of_model_sse": tot / max(float(
                      (wells["mse_anom_model"] * wells["n_months"]).sum()), 1e-12),
                  "frac_trend": float((d["mse_trend_model"] * n).sum()) / max(tot, 1e-12),
                  "frac_amplitude": float((d["mse_amplitude_model"] * n).sum())
                  / max(tot, 1e-12),
                  "frac_phase": float((d["mse_phase_model"] * n).sum()) / max(tot, 1e-12),
                  "median_sd_obs_m": float(d["sd_anom_obs_model"].median()),
                  "median_sd_model_m": float(d["sd_anom_model_model"].median()),
                  "median_sd_ratio": float(d["sd_ratio_model"].median()),
                  "median_corr_detrended": float(d["corr_detrended_model"].median()),
                  "median_corr_detrended_idw": float(d["corr_detrended_idw"].median()),
                  "median_trend_obs_m_yr": float(d["trend_obs_m_yr_model"].median()),
                  "median_trend_model_m_yr": float(d["trend_model_m_yr_model"].median())})
    return wells, pd.DataFrame(g)


# ---------------------------------------------------------------------------------------
# the fold hindcasts (torch + the data cache)
# ---------------------------------------------------------------------------------------
def fold_model_heads(run_dir: str, device: str = "cpu", log=print) -> dict:
    """Rebuild every fold member of ``run_dir`` and hindcast it from its kept-wells
    initial field -> ``{"model": (F, W, T) heads at every well's cell and layer (months
    1..T of the record, the gate's target months), "obs": (W, T), "xy", "layer",
    "zone", "sids", "fold_of": (W,) the fold that held each well out, "check": per-fold
    max |rebuilt - dumped| at its held-out wells}``."""
    import torch

    from .forward import N_LAYERS, build_model, hindcast_with_nudging, load_members, sw_hist
    from .inputs import input_options, load_twin_inputs
    from .zones import fan_zones

    torch.set_grad_enabled(False)
    theta_p = os.path.join(run_dir, "stage3_theta.json")
    members = load_members([theta_p, os.path.join(run_dir, "stage3_fold_thetas.json")])
    meta = members[0].meta
    inp = load_twin_inputs(**input_options(meta), verbose=False)
    with np.load(os.path.join(run_dir, PER_ENTRY)) as z:
        pe = {k: z[k] for k in z.files}
    W = len(inp.sids)
    fold_of = np.full(W, -1, dtype="int64")
    fold_of[pe["entry"]] = pe["fold"]
    if (fold_of < 0).any():
        raise ValueError("some wells are held out by no fold: the dump and the inputs "
                         "disagree on the well list")
    obs = inp.obs_h_filled[:, 1:]
    held_obs = obs[pe["entry"]]
    if not np.allclose(held_obs, pe["obs"], equal_nan=True, atol=1e-9):
        raise ValueError("the dump's held-out obs differ from the inputs' back-filled heads "
                         "(different well list or construction)")
    E = inp.E_total[:, 1:]
    R = inp.recharge_field[:, 1:]
    sw = sw_hist(inp)
    folds = [m for m in members if m.label.startswith("fold")]
    heads = np.full((len(folds), W, obs.shape[1]), np.nan)
    check = {}
    for mem in folds:
        f = int(mem.label[4:])
        t0 = time.perf_counter()
        keep = fold_of != f
        h0 = inp.initial_heads(0, n_layers=N_LAYERS, well_mask=keep)
        model, scalars, _ = build_model(inp.grid, mem, device)
        h = hindcast_with_nudging(model, scalars, inp, h0, E, R, 0.0, 0, sw=sw)
        heads[f] = h[inp.obs_layer, inp.obs_idx, 1:].cpu().numpy()
        sel = pe["fold"] == f
        check[f] = float(np.nanmax(np.abs(heads[f][pe["entry"][sel]] - pe["pred"][sel])))
        log(f"  fold {f}: hindcast {time.perf_counter() - t0:.1f}s, rebuilt vs dumped "
            f"held-out prediction max |diff| {check[f]:.2e} m")
    zb = [float(v) for v in str(meta.get("zone_boundaries", "205,182")).split(",")]
    cell_xy = inp.grid.centroids()[inp.obs_idx]
    zone = fan_zones(cell_xy, proximal_km=zb[0], distal_km=zb[1],
                     split_km=zb[2] if len(zb) > 2 else None)
    return {"model": heads, "obs": obs, "cell_xy": cell_xy, "layer": inp.obs_layer,
            "zone": zone, "sids": np.asarray(inp.sids), "fold_of": fold_of,
            "check": check, "per_entry": pe}


def evaluate(fm: dict, kriging: bool = True, power: float = IDW_POWER) -> dict:
    """Every method's held-out predictions, pooled in dump order, and their anomaly
    verdicts against the gate's IDW. ``fm`` is :func:`fold_model_heads`' result (or the
    cache ``--reuse`` reads back)."""
    pe = fm["per_entry"]
    obs, xy, lay = fm["obs"], fm["cell_xy"], fm["layer"]
    preds = {k: np.full_like(pe["pred"], np.nan) for k in METHODS}
    variograms, betas = {}, {}
    for f in np.unique(pe["fold"]):
        rows = np.flatnonzero(pe["fold"] == f)
        held = pe["entry"][rows]
        keep = np.flatnonzero(fm["fold_of"] != f)
        model = fm["model"][f]
        r = residual_anomalies(model[keep], obs[keep])
        a_obs = row_anomaly(obs[keep])
        preds["model"][rows] = pe["pred"][rows]
        preds["idw_heads"][rows] = pe["idw"][rows]
        preds["idw_anom"][rows] = idw_interp(xy[held], xy[keep], a_obs, power=power)
        preds["idw_anom_layer"][rows] = idw_correction(xy[held], xy[keep], a_obs,
                                                       power=power, src_layer=lay[keep],
                                                       tgt_layer=lay[held])
        preds["rk_idw"][rows] = pe["pred"][rows] + idw_correction(xy[held], xy[keep], r,
                                                                   power=power)
        preds["rk_idw_layer"][rows] = pe["pred"][rows] + idw_correction(
            xy[held], xy[keep], r, power=power, src_layer=lay[keep], tgt_layer=lay[held])
        beta = inner_beta(xy[keep], model[keep], obs[keep], power=power)
        betas[int(f)] = beta
        a_mod = row_anomaly(model[keep])
        preds["rk_beta"][rows] = preds["idw_anom"][rows] + beta * (
            row_anomaly(pe["pred"][rows]) - idw_interp(xy[held], xy[keep], a_mod, power=power))
        if kriging:
            vg = empirical_semivariogram(xy[keep], r)
            vp = fit_exponential(vg)
            variograms[int(f)] = {**vp, "bins": vg.to_dict("records")}
            preds["rk_sk"][rows] = pe["pred"][rows] + simple_kriging(xy[held], xy[keep], r,
                                                                      vp)
            vo = fit_exponential(empirical_semivariogram(xy[keep], a_obs))
            variograms[int(f)]["obs_anomaly"] = vo
            preds["sk_anom"][rows] = simple_kriging(xy[held], xy[keep], a_obs, vo)
    scores = {}
    for k, p in preds.items():
        if not np.isfinite(p).any():
            continue
        s = kfold_anomaly_scores(p, pe["idw"], pe["obs"])
        s["r2_abs"] = _r2(p, pe["obs"]) if k in ("model", "idw_heads") else float("nan")
        scores[k] = s
    return {"preds": preds, "scores": scores, "variograms": variograms, "betas": betas}


def site_groups(xy: np.ndarray) -> np.ndarray:
    """Integer label per row: rows at the same (cell-centroid) location share a label."""
    _, lab = np.unique(np.round(np.asarray(xy, dtype="float64"), 1), axis=0,
                       return_inverse=True)
    return lab.ravel()


def inner_beta(xy: np.ndarray, model: np.ndarray, obs: np.ndarray,
               power: float = IDW_POWER) -> float:
    """Weight of the model's local departure from its own interpolation, fitted on the
    fold's KEPT wells only by leave-one-location-out: for each kept location, ``z`` = its
    observed anomaly minus the IDW of the other kept wells' observed anomalies, ``x`` = the
    same for the model's anomalies; ``beta = sum(x z) / sum(x x)``, clipped to [0, 1].
    ``beta = 1`` is plain regression kriging with IDW, ``beta = 0`` is IDW. The kept wells
    were in the fold's fit, so the model is in-sample there: ``beta`` leans high, never on
    held-out data."""
    a_o, a_m = row_anomaly(obs), row_anomaly(model)
    lab = site_groups(xy)
    num = den = 0.0
    for g in np.unique(lab):
        t, s = lab == g, lab != g
        z = a_o[t] - idw_interp(xy[t], xy[s], a_o[s], power=power)
        x = a_m[t] - idw_interp(xy[t], xy[s], a_m[s], power=power)
        m = np.isfinite(z) & np.isfinite(x)
        num += float((x[m] * z[m]).sum())
        den += float((x[m] ** 2).sum())
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


HYBRID_CONTROLS = {"rk_idw": "idw_anom", "rk_idw_layer": "idw_anom_layer",
                   "rk_sk": "sk_anom", "rk_beta": "idw_anom"}


def bootstrap_margins(preds: dict, pe: dict, xy_entry: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, ref: np.ndarray | None = None) -> pd.DataFrame:
    """Anomaly-R2 margin of every method over the gate's IDW (or over ``ref``, a
    prediction array of the same shape), with a bootstrap over held-out LOCATIONS
    (co-located screens resampled together): median, 5th and 95th percentile, and the
    fraction of resamples with a positive margin."""
    mask = scored_mask(pe["pred"], pe["idw"], pe["obs"])
    ao = own_mean_anomaly(pe["obs"], mask)
    lab = site_groups(xy_entry)
    groups = [np.flatnonzero(lab == g) for g in np.unique(lab)]
    rng = np.random.default_rng(seed)
    ai = own_mean_anomaly(pe["idw"] if ref is None else ref, mask)
    am = {k: own_mean_anomaly(p, mask) for k, p in preds.items() if np.isfinite(p).any()}
    sse = {k: np.nansum((a - ao) ** 2, 1) for k, a in am.items()}
    sse_i, sst = np.nansum((ai - ao) ** 2, 1), np.nansum(ao ** 2, 1)
    out = {k: [] for k in am}
    for _ in range(n_boot):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        st = sst[rows].sum()
        for k in am:
            out[k].append((sse_i[rows].sum() - sse[k][rows].sum()) / st)
    return pd.DataFrame([{"method": k, "margin": float((sse_i.sum() - sse[k].sum())
                                                       / sst.sum()),
                          "boot_p05": float(np.percentile(v, 5)),
                          "boot_median": float(np.median(v)),
                          "boot_p95": float(np.percentile(v, 95)),
                          "frac_positive": float((np.asarray(v) > 0).mean())}
                         for k, v in out.items()])


def per_group_scores(preds: dict, pe: dict, zone_of_entry: np.ndarray,
                     zone_names: tuple[str, ...]) -> pd.DataFrame:
    """Pooled anomaly R2 of every method by zone and by layer (the gate's months)."""
    mask = scored_mask(pe["pred"], pe["idw"], pe["obs"])
    ao = own_mean_anomaly(pe["obs"], mask)
    rows = []
    groups = [("all", np.ones(len(ao), bool))]
    groups += [(f"zone={zone_names[z]}", zone_of_entry == z) for z in np.unique(zone_of_entry)]
    groups += [(f"layer={k + 1}", pe["layer"] == k) for k in np.unique(pe["layer"])]
    for name, g in groups:
        row = {"group": name, "n_wells": int(g.sum())}
        for k, p in preds.items():
            if np.isfinite(p).any():
                row[k] = _r2(own_mean_anomaly(p, mask)[g], ao[g])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# the display correction (part c)
# ---------------------------------------------------------------------------------------
CORRECTION_METHODS = ("idw", "idw_layer", "sk")


def head_field_correction(model_w: np.ndarray, obs: np.ndarray, well_layer: np.ndarray,
                          src_xy: np.ndarray, cell_xy: np.ndarray, n_layers: int,
                          method: str = "idw_layer", power: float = IDW_POWER) -> tuple:
    """``(L, A, T)`` correction field from the wells' residual anomalies ``(obs - mean) -
    (model - mean)`` (``model_w``/``obs`` ``(W, T)``), NaN-free (0 where no well has a
    residual that month). ``method``: ``idw`` (every layer pooled, the gate's IDW),
    ``idw_layer`` (layer k's cells from layer-k wells; all wells for a layer with none) or
    ``sk`` (simple kriging, every layer pooled, exponential covariance fitted to these
    residuals). Returns ``(correction, info)``."""
    if method not in CORRECTION_METHODS:
        raise ValueError(f"method must be one of {CORRECTION_METHODS}, got {method!r}")
    r = residual_anomalies(model_w, obs)
    info: dict = {"method": method, "n_wells": int(np.isfinite(r).any(1).sum())}
    if method == "idw_layer":
        out = np.zeros((n_layers, len(cell_xy), r.shape[1]))
        allw = idw_interp(cell_xy, src_xy, r, power=power)
        for k in range(n_layers):
            s = well_layer == k
            out[k] = idw_interp(cell_xy, src_xy[s], r[s], power=power) if s.any() else allw
            info[f"layer{k + 1}_wells"] = int(s.sum())
    else:
        if method == "sk":
            vp = fit_exponential(empirical_semivariogram(src_xy, r))
            info["variogram"] = vp
            c = simple_kriging(cell_xy, src_xy, r, vp)
        else:
            c = idw_interp(cell_xy, src_xy, r, power=power)
        out = np.broadcast_to(c, (n_layers, *c.shape)).copy()
    return np.where(np.isfinite(out), out, 0.0), info


def correct_head_field(heads, obs: np.ndarray, well_layer: np.ndarray,
                       well_idx: np.ndarray, cell_xy: np.ndarray,
                       method: str = "idw_layer", power: float = IDW_POWER):
    """``heads`` ``(L, A, T)`` (numpy or torch; months aligned with ``obs`` ``(W, T)``)
    plus :func:`head_field_correction` of the model's residuals at the wells (each well
    read at its cell and layer). At a well's own cell the corrected anomaly reproduces the
    observed one (IDW is exact at a source); the well's LEVEL stays the model's (a datum
    or level offset is not interpolated). Returns ``(corrected, correction, info)``, the
    first in the type of ``heads``."""
    is_torch = not isinstance(heads, np.ndarray)
    h = heads.detach().cpu().numpy() if is_torch else np.asarray(heads, dtype="float64")
    corr, info = head_field_correction(h[well_layer, well_idx, :], obs, well_layer,
                                       cell_xy[well_idx], cell_xy, h.shape[0],
                                       method=method, power=power)
    out = h + corr
    if is_torch:
        import torch

        out = torch.as_tensor(out, dtype=heads.dtype, device=heads.device)
    return out, corr, info


def correct_forward(fwd_npz: str, theta_json: str, out_npz: str,
                    method: str = "sk", log=print) -> dict:
    """Opt-in display correction of a finished ``forward`` run (part c): a copy of
    ``fwd_npz`` whose ``heads_mean`` hindcast months (0 .. origin) carry the residual
    correction of every scenario's ensemble-mean heads against the observed (back-filled,
    as the gate scored) heads of every well. Projection months are left as they are (no
    observation, no correction); ``heads_std`` and every subsidence array are untouched --
    the compaction column is never driven by corrected heads. The correction is linear in
    the model heads, so correcting the ensemble mean equals the mean of the corrected
    members. Adds ``heads_correction`` ``(S, L, A, T_obs)`` float32 and
    ``residual_correction`` (JSON)."""
    from .inputs import input_options, load_twin_inputs

    with open(theta_json) as fh:
        meta = json.load(fh).get("meta", {})
    inp = load_twin_inputs(**input_options(meta), verbose=False)
    with np.load(fwd_npz, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    T = len(inp.dates)
    fdates = pd.DatetimeIndex(pd.to_datetime(data["dates"][:T]))
    if not (fdates == inp.dates).all():
        raise ValueError("the forward run's first months are not the inputs' record")
    hm = data["heads_mean"].astype("float64")
    cell_xy = inp.grid.centroids()
    if hm.shape[2] != len(cell_xy):
        raise ValueError(f"{fwd_npz}: {hm.shape[2]} cells, the inputs' grid has "
                         f"{len(cell_xy)}")
    corr_all = np.zeros(hm.shape[:3] + (T,), dtype="float32")
    infos = []
    for s in range(hm.shape[0]):
        new, corr, info = correct_head_field(hm[s, :, :, :T], inp.obs_h_filled,
                                             inp.obs_layer, inp.obs_idx, cell_xy,
                                             method=method)
        hm[s, :, :, :T] = new
        corr_all[s] = corr
        infos.append(info)
    jump = float(np.abs(corr_all[:, :, :, T - 1]).mean())
    rec = {"method": method, "months": [0, T - 1], "source": "obs_h_filled",
           "fwd": fwd_npz, "theta": theta_json, "mean_abs_correction_m": float(
               np.abs(corr_all).mean()), "mean_abs_correction_at_origin_m": jump,
           "validated": "residual_kriging k-fold (rk_summary.json)", "info": infos[0],
           "note": "display only: heads_std and subsidence are the uncorrected run's"}
    data["heads_mean"] = hm.astype("float32")
    data["heads_correction"] = corr_all
    data["residual_correction"] = np.array(json.dumps(_jsonable(rec)))
    np.savez_compressed(out_npz, **data)
    log(f"wrote {out_npz}: {method} correction on months 0..{T - 1}, mean |correction| "
        f"{rec['mean_abs_correction_m']:.2f} m ({jump:.2f} m at the origin, where the "
        "uncorrected projection takes over)")
    return rec


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main_correct(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="residual_kriging correct",
                                 description="opt-in residual correction of a forward "
                                 "run's displayed hindcast heads")
    ap.add_argument("forward_npz", help="a forward run's <out>.npz")
    ap.add_argument("--theta", required=True, help="the run's in-sample stage3_theta.json")
    ap.add_argument("--out", required=True, help="new .npz path (never the input)")
    ap.add_argument("--method", choices=CORRECTION_METHODS, default="sk",
                    help="sk (default: the best k-fold hybrid, +0.635 vs gate IDW +0.605), "
                         "idw_layer (+0.622) or idw (+0.599, below the gate IDW)")
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.forward_npz):
        raise SystemExit("--out must differ from the forward run's own file")
    correct_forward(args.forward_npz, args.theta, args.out, method=args.method)


def main(argv=None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "correct":
        return main_correct(argv[1:])
    ap = argparse.ArgumentParser(description="regression kriging at k-fold held-out wells")
    ap.add_argument("run", help="k-fold run dir with stage3_theta.json, "
                                "stage3_fold_thetas.json and stage3_per_entry.npz")
    ap.add_argument("--out", required=True, help="new output directory (never the run's)")
    ap.add_argument("--no-kriging", action="store_true", help="IDW corrections only")
    ap.add_argument("--reuse", action="store_true",
                    help="read the fold hindcasts back from <out>/fold_model_heads.npz")
    ap.add_argument("--power", type=float, default=IDW_POWER)
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.run):
        raise SystemExit("--out must be a new directory, not the run's own")
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "fold_model_heads.npz")
    from .zones import zone_names as _zn

    if args.reuse and os.path.exists(cache):
        with np.load(cache) as z:
            fm = {k: z[k] for k in z.files if not k.startswith("pe_")}
            fm["per_entry"] = {k[3:]: z[k] for k in z.files if k.startswith("pe_")}
        fm["check"] = json.loads(str(fm.pop("check_json")))
    else:
        fm = fold_model_heads(args.run)
        np.savez_compressed(cache, **{k: v for k, v in fm.items()
                                      if k not in ("per_entry", "check")},
                            check_json=json.dumps(fm["check"]),
                            **{f"pe_{k}": v for k, v in fm["per_entry"].items()})
    pe = fm["per_entry"]
    znames = _zn(int(fm["zone"].max()) + 1 if fm["zone"].max() >= 3 else 3)
    zone_e = fm["zone"][pe["entry"]]
    wells, groups = diagnose(pe, zone_e, znames)
    wells.insert(1, "sid", fm["sids"][pe["entry"]])
    wells.to_csv(os.path.join(args.out, "rk_diagnosis_wells.csv"), index=False)
    groups.to_csv(os.path.join(args.out, "rk_diagnosis_groups.csv"), index=False)
    res = evaluate(fm, kriging=not args.no_kriging, power=args.power)
    pg = per_group_scores(res["preds"], pe, zone_e, znames)
    pg.to_csv(os.path.join(args.out, "rk_scores_by_group.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "rk_per_entry.npz"),
                        **{f"pred_{k}": v for k, v in res["preds"].items()},
                        obs=pe["obs"], idw=pe["idw"], entry=pe["entry"], fold=pe["fold"])
    boot = bootstrap_margins(res["preds"], pe, fm["cell_xy"][pe["entry"]])
    boot.to_csv(os.path.join(args.out, "rk_bootstrap.csv"), index=False)
    print(boot.round(3).to_string(index=False))
    # each hybrid against the model-free interpolator it is built on: the model's own share
    vs = []
    for h, c in HYBRID_CONTROLS.items():
        if np.isfinite(res["preds"][h]).any() and np.isfinite(res["preds"][c]).any():
            b = bootstrap_margins({h: res["preds"][h]}, pe, fm["cell_xy"][pe["entry"]],
                                  ref=res["preds"][c])
            vs.append(b.assign(control=c))
    if vs:
        vs = pd.concat(vs, ignore_index=True)
        vs.to_csv(os.path.join(args.out, "rk_bootstrap_vs_control.csv"), index=False)
        print(vs.round(3).to_string(index=False))
    summary = {"run": args.run, "power": args.power, "betas": res["betas"],
               "bootstrap": boot.to_dict("records"),
               "bootstrap_vs_control": (vs.to_dict("records")
                                        if isinstance(vs, pd.DataFrame) else []),
               "max_abs_diff_vs_dump_m": fm["check"], "scores": res["scores"],
               "variograms": {k: {kk: vv for kk, vv in v.items() if kk != "bins"}
                              for k, v in res["variograms"].items()}}
    with open(os.path.join(args.out, "rk_summary.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    pd.set_option("display.width", 200)
    print(groups.round(3).to_string(index=False))
    print(pg.round(3).to_string(index=False))
    for k, s in res["scores"].items():
        print(f"{k:13s} anomaly R2 {s['r2_anom_kfold']:+.3f} (IDW {s['r2_anom_idw']:+.3f}, "
              f"margin {s['margin_anom_kfold']:+.3f} {s['verdict_anom_kfold']}); per-well "
              f"median {s['r2_anom_well_median_kfold']:+.3f}")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

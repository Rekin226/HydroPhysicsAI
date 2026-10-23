"""WP P4m main run: engine checks EC1-EC4, T1 (skill change), T2 (heads-accepted scale of the
metered series, shape comparison, response/conversion sensitivity T2c) and T3 (E10-style
check), as declared in ``P4m.metered_diagnostics.v1``.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_run

Writes ``results/dataprep/reports/P4m_evidence/stage/run/`` (results.json, arrays.npz,
run.log) and records its looks. The theta-refit emulation (T2b) is ``p4m_refit``.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p4_changhua import engine, engine_tlm, looks_p4m  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402

OUT = L.STAGE / "run"
SENS_STEP = 0.1


# ============================================================================ helpers
def response_amplitude(P0: np.ndarray, P1: np.ndarray, wsel: np.ndarray, tm: np.ndarray) -> dict:
    """rms over the wells of ``wsel`` and the scored steps of the anomaly (per-well mean
    removed) of the head response P1 - P0, and the mean response (m)."""
    d = (P1 - P0)[wsel][:, tm]
    an = d - d.mean(axis=1, keepdims=True)
    return {"A_resp_m": float(np.sqrt(np.mean(an ** 2))), "mean_resp_m": float(d.mean())}


def scale_family(S, obs_m, base_extra, grid, elig, ch_el, tag):
    """Exact stepwise rollouts at E_extra = s x base_extra: objective curves, amplitude ratio,
    s_LS (all wells and Changhua wells), s_AM; per-well SSE curves for the bootstrap."""
    from scipy.optimize import minimize_scalar

    cache: dict[float, np.ndarray] = {}

    def pred(s: float) -> np.ndarray:
        key = round(float(s), 6)
        if key not in cache:
            cache[key] = L.at_wells(S, L.rollout(S, E_extra=s * base_extra if s else None))
        return cache[key]

    def J(s, sel):
        return float(np.nansum(L.anomaly_sse(obs_m, pred(s), sel)))

    curves = np.stack([L.anomaly_sse(obs_m, pred(s), elig) for s in grid], axis=1)
    ar = [L.amp_ratio_group(obs_m, pred(s), ch_el) for s in grid]
    res = {"grid": list(grid), "J_all": [J(s, elig) for s in grid],
           "J_changhua": [J(s, ch_el) for s in grid], "amp_ratio_changhua": ar}
    for name, sel in (("all", elig), ("changhua", ch_el)):
        vals = res["J_all" if name == "all" else "J_changhua"]
        k = int(np.argmin(vals))
        lo = grid[max(k - 1, 0)]
        hi = grid[min(k + 1, len(grid) - 1)]
        if lo == hi:
            s_hat = lo
        else:
            r = minimize_scalar(lambda s, sel=sel: J(s, sel), bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-3})
            s_hat = float(r.x)
        res[f"s_LS_{name}"] = s_hat
        res[f"s_LS_{name}_at_bound"] = bool(s_hat >= grid[-1] - 1e-3)
        res[f"J_{name}_at_s_LS"] = J(s_hat, sel)
    # amplitude-matched: first crossing of 1
    if ar[0] >= 1.0:
        s_am, how = 0.0, "amplitude ratio >= 1 at s = 0"
    else:
        s_am, how = None, "never reaches 1 on [0, 3]"
        for i in range(1, len(grid)):
            if ar[i] >= 1.0:
                a, b = grid[i - 1], grid[i]
                for _ in range(14):
                    m = 0.5 * (a + b)
                    if L.amp_ratio_group(obs_m, pred(m), ch_el) >= 1.0:
                        b = m
                    else:
                        a = m
                s_am, how = 0.5 * (a + b), "bisection between grid points"
                break
    res["s_AM_changhua"] = s_am
    res["s_AM_how"] = how
    res["n_rollouts"] = len(cache)
    L.log(f"{tag}: s_LS all {res['s_LS_all']:.4f}, Changhua {res['s_LS_changhua']:.4f}; "
          f"s_AM {s_am}; AR(0) {ar[0]:.3f}, AR(1) {ar[grid.index(1.0)]:.3f}")
    return res, curves, cache


def boot_s(curves: np.ndarray, grid, cl, gsel) -> dict:
    """Cluster bootstrap of argmin_s of the summed per-well SSE curves (cubic spline)."""
    from scipy.interpolate import CubicSpline

    fine = np.linspace(grid[0], grid[-1], 3001)
    ok = gsel & np.all(np.isfinite(curves), axis=1)

    def stat(idx):
        c = curves[idx].sum(axis=0)
        return float(fine[np.argmin(CubicSpline(grid, c)(fine))])

    est = stat(np.flatnonzero(ok))
    return {"spline_estimate": est, **L.boot_interval(stat, cl, ok)}


def ols_free_datum(X: np.ndarray, r: np.ndarray, wsel: np.ndarray, *, detrend: bool = False,
                   idx_rows=None):
    """OLS of r (W, T) on X (W, T, P) with a free per-well datum (and linear trend), over the
    finite entries of r for wells in wsel (or the multiset idx_rows). Returns coefficients."""
    rows_X, rows_y = [], []
    wells = np.flatnonzero(wsel) if idx_rows is None else idx_rows
    for w in wells:
        m = np.isfinite(r[w])
        if m.sum() < L.MIN_N:
            continue
        y = r[w, m]
        x = X[w, m]
        if detrend:
            t = np.flatnonzero(m).astype(float)
            Z = np.column_stack([np.ones_like(t), t - t.mean()])
            proj = Z @ np.linalg.lstsq(Z, np.column_stack([y, x]), rcond=None)[0]
            y = y - proj[:, 0]
            x = x - proj[:, 1:]
        else:
            y = y - y.mean()
            x = x - x.mean(axis=0, keepdims=True)
        rows_X.append(x)
        rows_y.append(y)
    A = np.concatenate(rows_X)
    b = np.concatenate(rows_y)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def gls_free_datum(X: np.ndarray, r: np.ndarray, wsel: np.ndarray, phi: float):
    """AR(1)-whitened least squares with a free per-well datum (datum column whitened and
    projected out per well; the pilot convention)."""
    rows_X, rows_y = [], []
    for w in np.flatnonzero(wsel):
        m = np.isfinite(r[w])
        if m.sum() < L.MIN_N:
            continue
        t = np.flatnonzero(m)
        Z = np.column_stack([r[w, m], np.ones(t.size), X[w, m]])
        Wz = np.empty_like(Z)
        Wz[0] = Z[0] * math.sqrt(1 - phi ** 2)
        for i in range(1, t.size):
            Wz[i] = Z[i] - phi * Z[i - 1] if t[i] == t[i - 1] + 1 else Z[i] * math.sqrt(
                1 - phi ** 2)
        d = Wz[:, 1:2]
        P = Wz - d @ (d.T @ Wz) / float(d.T @ d)
        rows_y.append(P[:, 0])
        rows_X.append(P[:, 2:])
    return np.linalg.lstsq(np.concatenate(rows_X), np.concatenate(rows_y), rcond=None)[0]


def lag1_phi(r: np.ndarray, wsel: np.ndarray) -> float:
    num = den = 0.0
    for w in np.flatnonzero(wsel):
        x = r[w] - np.nanmean(r[w])
        a, b = x[1:], x[:-1]
        ok = np.isfinite(a) & np.isfinite(b)
        num += float((a[ok] * b[ok]).sum())
        den += float(np.nansum(x ** 2))
    return num / den


def channel_fit(X, r, wsel, cl, *, detrend=False, phi=None, names=("mean", "seasonal",
                                                                     "anomaly")):
    est = ols_free_datum(X, r, wsel, detrend=detrend)
    boots = []
    for idx in L.boot_indices(cl, wsel):
        try:
            boots.append(ols_free_datum(X, r, wsel, detrend=detrend, idx_rows=idx))
        except (np.linalg.LinAlgError, ValueError):
            continue
    boots = np.asarray(boots)
    out = {"method": "OLS, free per-well datum" + (" + linear trend" if detrend else ""),
           "n_boot": int(len(boots)), "label": "cluster bootstrap (10 km), UNCALIBRATED"}
    for k, nm in enumerate(names):
        out[nm] = {"theta": float(est[k]), "p05": float(np.percentile(boots[:, k], 5)),
                   "p95": float(np.percentile(boots[:, k], 95))}
    if phi is not None:
        g = gls_free_datum(X, r, wsel, phi)
        out["gls_ar1"] = {"phi": phi, **{nm: float(g[k]) for k, nm in enumerate(names)}}
    return out


def per_well_transient_trend(resid: np.ndarray, steps: np.ndarray, tm: np.ndarray):
    early = steps <= "2013-12"
    late = (steps >= "2015-01") & (steps <= "2022-12")
    W = resid.shape[0]
    tr, sl, tt = np.full(W, np.nan), np.full(W, np.nan), np.full(W, np.nan)
    yrs = np.array([int(s[:4]) + (int(s[5:7]) - 0.5) / 12 for s in steps])
    for w in range(W):
        e, la = np.isfinite(resid[w]) & early, np.isfinite(resid[w]) & late
        if e.sum() >= 3 and la.sum() >= L.MIN_N:
            tr[w] = resid[w, e].mean() - resid[w, la].mean()
        m = np.isfinite(resid[w]) & tm
        if m.sum() >= L.MIN_N:
            x, y = yrs[m], resid[w, m]
            xc = x - x.mean()
            b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
            res = y - y.mean() - b * xc
            se = math.sqrt(float((res ** 2).sum()) / (m.sum() - 2) / float((xc ** 2).sum()))
            sl[w], tt[w] = b, b / se if se > 0 else np.nan
    return tr, sl, tt


def summarise_wellstat(v, cl, gsel, name) -> dict:
    ok = gsel & np.isfinite(v)
    out = {"stat": name, "n_wells": int(ok.sum()), "mean": float(np.mean(v[ok])),
           "median": float(np.median(v[ok]))}
    out.update(L.boot_interval(lambda idx: float(np.mean(v[idx])), cl, ok))
    return out


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    R: dict = {"wp": "P4m", "prereg": L.PRE_DIAG, "lib_version": L.__version__,
               "created": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    S = L.setup()
    tw = S["tw"]
    tm = L.step_mask(S)
    obs_m = L.masked_obs(S, tm)
    elig = L.eligible(obs_m)
    G = L.groups(S)
    ch_el = G["CH"] & elig
    cl = L.clusters(S)
    steps = S["steps"]
    R["setup"] = {"theta_sha256": L.THETA_SHA, "n_wells": int(len(S["sids"])),
                  "n_steps": int(S["E"].shape[1]), "steps": [steps[0], steps[-1]],
                  "scored": [L.SCORE_FIRST, L.SCORE_LAST], "eligible_wells": int(elig.sum()),
                  "eligible_changhua": int(ch_el.sum()),
                  "eligible_yunlin": int((G["YL"] & elig).sum()),
                  "n_clusters": int(np.unique(cl).size),
                  "torch_threads": int(os.environ.get("OMP_NUM_THREADS", "1"))}

    # ------------------------------------------------------------- EC1-EC4
    H0, Q0 = L.rollout(S, return_rates=True)
    ec1 = float(np.abs(H0 - S["H_hindcast_v2"]).max())
    Ech, Ech_full, info1 = L.changhua_forcing(S)
    Ech_nd, Ech_nd_full, info2 = L.changhua_forcing(S, nodomestic=True)
    ref_all = L.c0_county_totals("preorigin", "model_design", ["changhua", "chiayi"],
                                 "in_grid", "all", "2012-01", "2022-12")
    ref_nd = L.c0_county_totals("preorigin", "model_design", ["changhua", "chiayi"],
                                "in_grid", "all_nodomestic", "2012-01", "2022-12")
    ec2 = {"arm_A1_kwh": float(Ech_full.sum()), "c0_table_all_kwh": ref_all,
           "rel_A1": abs(float(Ech_full.sum()) - ref_all) / ref_all,
           "arm_A2_kwh": float(Ech_nd_full.sum()), "c0_table_nodomestic_kwh": ref_nd,
           "rel_A2": abs(float(Ech_nd_full.sum()) - ref_nd) / ref_nd,
           "GWh_per_yr_A1": float(Ech_full.sum()) / 11 / 1e6,
           "GWh_per_yr_A2": float(Ech_nd_full.sum()) / 11 / 1e6,
           "c0_flags_A1": info1["flags"], "guard_notes": info1["notes"]}
    H1, Q1 = L.rollout(S, E_extra=Ech, return_rates=True)
    H2 = L.rollout(S, E_extra=Ech_nd)
    t0 = time.perf_counter()
    vt = engine.verify_against_torch(tw, S["sc"], S["h0"], S["E"], S["R"], E_extra=Ech,
                                     H_sparse=H1)
    ec3 = {k: v for k, v in vt.items() if k != "H_torch"}
    ec3["t_s"] = round(time.perf_counter() - t0, 1)
    ec4 = {"A1": engine_tlm.clipped_lift_count(tw, H1),
           "A0": engine_tlm.clipped_lift_count(tw, H0)}
    # Yunlin forcing of the model vs C0 v3 Yunlin (information)
    E_yl_c0, _, _ = L.c0_cells("preorigin", "model_design", counties=("yunlin",),
                               first="2012-02", last="2022-12", grid_hash_expected=S["grid_hash"])
    tot_m, tot_c = S["E"].sum(axis=0), E_yl_c0.sum(axis=0)
    rel = np.abs(tot_c - tot_m) / tot_m
    R["engine_checks"] = {
        "EC1_base_vs_engine_v2_max_abs_dh_m": ec1, "EC1_pass": bool(ec1 <= 1e-9),
        "EC2": ec2, "EC2_pass": bool(ec2["rel_A1"] <= 1e-9 and ec2["rel_A2"] <= 1e-9),
        "EC3_sparse_vs_torch_A1": ec3, "EC3_pass": bool(ec3["max_abs_dh_m"] <= 1e-5),
        "EC4_lift_clipped": ec4,
        "info_yunlin_model_forcing_vs_c0v3": {
            "max_rel_monthly_total_2012_02_2022_11": float(rel[:-1].max()),
            "rel_2022_12": float((tot_c[-1] - tot_m[-1]) / tot_m[-1])},
        "abstraction_m3_per_day_mean": {
            "A0_total": float(Q0.sum(axis=0).mean()),
            "A1_total": float(Q1.sum(axis=0).mean()),
            "A1_added": float((Q1 - Q0).sum(axis=0).mean()),
            "A1_added_note": "includes the lift feedback of the added energy on the base "
                             "conversion (base cells' rates change too)"}}
    L.log(f"EC1 {ec1:.2e} m; EC2 rel {ec2['rel_A1']:.1e}/{ec2['rel_A2']:.1e}; EC3 "
          f"{ec3['max_abs_dh_m']:.2e} m; EC4 clipped {ec4['A1']['n_clipped']}")
    looks_p4m.record(["p4m.EC.numerical"], ids=None, window="2012-01..2022-12",
                     description="", purpose="verify", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_run:EC:{R['created']}]",
                     note="EC1-EC4: sparse vs engine/v2 and torch on A1 forcing; forcing totals "
                          "vs C0 tables", none_reason=looks_p4m.NUMERICAL_NONE)

    # ------------------------------------------------------------- T1
    _insample_look(S, R, "p4m.T1.insample", "T1",
                   "T1 skill change of arms A0/A1/A2, observed months 2013-01..2022-12")
    P = {"A0": L.at_wells(S, H0), "A1": L.at_wells(S, H1), "A2": L.at_wells(S, H2)}
    pw = {k: L.per_well_stats(obs_m, v) for k, v in P.items()}
    T1 = {"groups": {}, "differences": {}}
    for g, gm in G.items():
        T1["groups"][g] = {arm: L.group_metrics(pw[arm], gm) for arm in P}
    for g in ("CH", "YL", "all"):
        T1["differences"][g] = {"A1_minus_A0": L.drmse_boot(pw["A1"], pw["A0"], cl, G[g]),
                                "A2_minus_A0": L.drmse_boot(pw["A2"], pw["A0"], cl, G[g])}
        for arm in ("A1", "A2"):
            T1["differences"][g][f"{arm}_relative_rmse_change"] = (
                T1["groups"][g][arm]["anomaly_rmse"] / T1["groups"][g]["A0"]["anomaly_rmse"] - 1)
    T1["harness_agreement_max_abs"] = max(L.check_against_harness(obs_m, P[a], G[g])
                                          for a in P for g in ("CH", "YL", "all"))
    T1["response_at_wells"] = {g: {arm: response_amplitude(P["A0"], P[arm], G[g] & elig, tm)
                                   for arm in ("A1", "A2")} for g in ("CH", "YL")}
    T1["mean_residual_change_m"] = {
        g: {arm: float(np.nanmean(pw[arm]["mean_resid"][G[g] & elig]
                                  - pw["A0"]["mean_resid"][G[g] & elig])) for arm in ("A1", "A2")}
        for g in ("CH", "YL")}
    R["T1"] = T1
    c = T1["groups"]["CH"]
    L.log(f"T1 Changhua anomaly RMSE A0 {c['A0']['anomaly_rmse']:.4f} A1 "
          f"{c['A1']['anomaly_rmse']:.4f} A2 {c['A2']['anomaly_rmse']:.4f}; amp "
          f"{c['A0']['amplitude_ratio']:.3f}/{c['A1']['amplitude_ratio']:.3f}; Yunlin "
          f"{T1['groups']['YL']['A0']['anomaly_rmse']:.4f} -> "
          f"{T1['groups']['YL']['A1']['anomaly_rmse']:.4f}")

    # ------------------------------------------------------------- T2 scale (true series)
    _insample_look(S, R, "p4m.T2.insample", "T2",
                   "T2 heads-accepted scale of the metered series, T2a Yunlin-shape "
                   "comparison, T2c response/conversion sensitivity; 2013-01..2022-12")
    grid = list(L.S_GRID)
    T2, curves, cache = scale_family(S, obs_m, Ech, grid, elig, ch_el, "T2 metered")
    T2["boot_s_LS_all"] = boot_s(curves, grid, cl, elig)
    T2["boot_s_LS_changhua"] = boot_s(curves, grid, cl, ch_el)
    T2["far_below_1"] = bool(T2["s_LS_all"] < 0.5 or (T2["s_AM_changhua"] is not None
                                                         and T2["s_AM_changhua"] < 0.67))
    # first-order cross-checks with the tangent-linear response about A0
    Es0 = tw.spread(S["E"])
    X1 = L.tlm_response(S, H0, Es0, Ech)                                  # (W, T)
    r0 = obs_m - P["A0"]
    s_ols = ols_free_datum(X1[..., None], r0, elig)[0]
    sw = engine.solve_with_datum(X1[elig][..., None], r0[elig])
    nonlin = float(np.abs((P["A1"] - P["A0"]) - X1)[elig][:, tm].max())
    T2["first_order"] = {"s_ols_free_datum": float(s_ols),
                         "s_fit_flow_datum_prior": float(sw["theta"][0]),
                         "max_abs_stepwise_minus_tlm_at_s1_m": nonlin}
    # A2 (no domestic) scales
    T2_nd, curves_nd, _ = scale_family(S, obs_m, Ech_nd, grid, elig, ch_el, "T2 nodomestic")
    # T2a shape: Yunlin-shaped series with the same per-cell mean
    shape = S["E"].sum(axis=0) / S["E"].sum(axis=0).mean()
    E_ys = Ech.mean(axis=1, keepdims=True) * shape[None, :]
    T2_ys, curves_ys, _ = scale_family(S, obs_m, E_ys, grid, elig, ch_el, "T2a Yunlin shape")
    months = S["E_months"]
    cvs = {"changhua_chiayi_all_classes": L.seasonal_cv(Ech_full.sum(axis=0), months),
           "changhua_chiayi_nodomestic": L.seasonal_cv(Ech_nd_full.sum(axis=0), months),
           "yunlin_model_all_classes": L.seasonal_cv(
               np.stack([S["inp"].E_by_class[k] for k in S["inp"].E_by_class]).sum(
                   axis=(0, 1)), months),
           "yunlin_model_irrigation": L.seasonal_cv(
               S["inp"].E_by_class["irrigation"].sum(axis=0), months)}
    ch_irr = _class_series(S, "irrigation")
    cvs["changhua_chiayi_irrigation"] = L.seasonal_cv(ch_irr, months)
    cvs["ratio_irrigation_changhua_over_yunlin"] = (cvs["changhua_chiayi_irrigation"]
                                                    / cvs["yunlin_model_irrigation"])
    cvs["ratio_all_changhua_over_yunlin"] = (cvs["changhua_chiayi_all_classes"]
                                             / cvs["yunlin_model_all_classes"])
    cvs["yunlin_shape_series_cv"] = L.seasonal_cv(E_ys.sum(axis=0), months[1:])
    R["T2"] = {"metered": T2, "metered_nodomestic": T2_nd, "yunlin_shape": T2_ys,
               "seasonal_cv": cvs}
    np.savez_compressed(OUT / "scale_curves.npz", grid=np.array(grid), curves=curves,
                        curves_nodomestic=curves_nd, curves_yunlin_shape=curves_ys,
                        eligible=elig, changhua=G["CH"])

    # ------------------------------------------------------------- T2c sensitivity
    if not args.skip_sensitivity:
        R["T2c"] = sensitivity(S, Ech, obs_m, tm, elig, G, T2)
    # ------------------------------------------------------------- T3 (E10)
    _insample_look(S, R, "p4m.T3.insample", "T3",
                   "T3 E10-style check: transient window 2012-02..2013-12, trends and "
                   "channels 2013-01..2022-12 (V2 2014-01..), spin-up IC variant")
    R["T3"] = e10(S, H0, H1, Ech, obs_m, tm, elig, G, cl, P)
    np.savez_compressed(OUT / "arrays.npz", sids=S["sids"], steps=steps,
                        well_county=S["well_county"], well_zone=S["well_zone"],
                        obs_layer=S["ol"], P_A0=P["A0"], P_A1=P["A1"], P_A2=P["A2"],
                        obs=S["obs"], eligible=elig, clusters=cl, well_xy=S["well_xy"],
                        obs_idx=S["oi"],
                        E_ch_cells=Ech, E_ch_nodomestic_cells=Ech_nd,
                        Q_A0=Q0, Q_A1=Q1, H_A1_layer1=H1[1], H_A0_layer1=H0[1])
    R["t_total_s"] = round(time.perf_counter() - t_all, 1)
    L.jdump(R, OUT / "results.json")
    L.log(f"done in {R['t_total_s']}s -> {OUT}")
    return 0


def _insample_look(S, R, unit: str, stage: str, note: str) -> None:
    """Record the look when the stage starts (conservative: a crash later leaves it logged)."""
    looks_p4m.record([unit], ids=list(S["sids"]), window="2012-01..2022-12",
                     description="heads of the 158 calibration wells (build_head_field), "
                                 "observed months: in-sample diagnostics of the frozen cwa "
                                 "theta with metered Changhua forcing (the theta's fit data)",
                     purpose="insample_diagnostic", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_run:{stage}:{R['created']}]", note=note)


def _class_series(S, cls: str) -> np.ndarray:
    p = L.c0_open("preorigin", "model_design")
    with np.load(p.path("cells_by_class.npz"), allow_pickle=False) as z:
        dates = [str(x) for x in z["dates"]]
        i0, i1 = dates.index("2012-01"), dates.index("2022-12") + 1
        c = [str(x) for x in z["classes"]].index(cls)
        cty = [str(x) for x in z["counties"]]
        v = z["value_by_county"]
        return (v[cty.index("changhua"), c, :, i0:i1].sum(axis=0)
                + v[cty.index("chiayi"), c, :, i0:i1].sum(axis=0))


def sensitivity(S, Ech, obs_m, tm, elig, G, T2) -> dict:
    """T2c: d ln A_resp / d theta_j for every free entry, log_eta and log_head_extra."""
    from hydrophysics.twin.forward import Member

    t0 = time.perf_counter()
    member, meta = S["member"], S["meta"]
    entries = engine.theta_vector(member.theta, meta)
    pinned = {f"{p.name}[{p.index}]" for p in engine.bound_pinned(member.theta, meta)}
    chsel = G["CH"] & elig
    base0 = L.at_wells(S, L.rollout(S))
    base1 = L.at_wells(S, L.rollout(S, E_extra=Ech))
    A_hat = response_amplitude(base0, base1, chsel, tm)
    rm_hat = {a: float(np.sqrt(np.nanmean(L.per_well_stats(obs_m, p)["mse"][chsel])))
              for a, p in (("A0", base0), ("A1", base1))}

    def run_with(tw_p):
        p0 = L.at_wells(S, tw_p.rollout(S["h0"], E=S["E"], R=S["R"], apex_from=S["h0"]))
        p1 = L.at_wells(S, tw_p.rollout(S["h0"], E=S["E"], R=S["R"], E_extra=Ech,
                                        apex_from=S["h0"]))
        a = response_amplitude(p0, p1, chsel, tm)
        rm = {k: float(np.sqrt(np.nanmean(L.per_well_stats(obs_m, p)["mse"][chsel])))
              for k, p in (("A0", p0), ("A1", p1))}
        return a, rm

    rows = []
    targets = {"s_AM": T2.get("s_AM_changhua"), "s_LS_all": T2.get("s_LS_all")}
    specs = [(e["key"], e["index"], e["name"], e["value"], e["lo"], e["hi"]) for e in entries]
    specs += [("log_eta", 0, "log_eta[0] (fixed conversion)", math.log(S["tw"].eta), None,
               None),
              ("log_head_extra", 0, "log_head_extra[0] (fixed conversion)",
               math.log(S["tw"].hx), None, None)]
    for key, idx, name, val, lo, hi in specs:
        res = {}
        for sgn in (+1, -1):
            v = val + sgn * SENS_STEP
            if key in ("log_eta", "log_head_extra"):
                tw_p, _, _ = L.build_twin(member, S["grid"], S["ge"])
                if key == "log_eta":
                    tw_p.eta = math.exp(v)
                else:
                    tw_p.hx = math.exp(v)
            else:
                th = engine.perturb_theta(member.theta, key, idx, v)
                tw_p, _, _ = L.build_twin(Member(label="sens", theta=th, meta=meta), S["grid"],
                                          S["ge"])
            res[sgn] = run_with(tw_p)
        e_j = (math.log(res[1][0]["A_resp_m"]) - math.log(res[-1][0]["A_resp_m"])) / (
            2 * SENS_STEP)
        row = {"name": name, "value": val, "lo": lo, "hi": hi, "pinned": name in pinned,
               "dlnA_dtheta": e_j,
               "dmean_resp_dtheta_m": (res[1][0]["mean_resp_m"] - res[-1][0]["mean_resp_m"])
               / (2 * SENS_STEP),
               "d_rmse_A1_dtheta": (res[1][1]["A1"] - res[-1][1]["A1"]) / (2 * SENS_STEP),
               "d_rmse_A0_dtheta": (res[1][1]["A0"] - res[-1][1]["A0"]) / (2 * SENS_STEP)}
        for tname, s_t in targets.items():
            if s_t is None or s_t <= 0 or abs(e_j) < 1e-6:
                row[f"dtheta_to_scale_by_{tname}"] = None
                continue
            dth = math.log(s_t) / e_j
            new = val + dth
            row[f"dtheta_to_scale_by_{tname}"] = dth
            row[f"flag_{tname}"] = bool(abs(dth) > 2 or (lo is not None and new < lo)
                                        or (hi is not None and new > hi))
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["dlnA_dtheta"]))
    L.log(f"T2c sensitivity: {len(rows)} entries in {time.perf_counter() - t0:.1f}s; top "
          + ", ".join(f"{r['name']} {r['dlnA_dtheta']:+.3f}" for r in rows[:5]))
    return {"A_resp_theta_hat": A_hat, "changhua_rmse_theta_hat": rm_hat, "step": SENS_STEP,
            "targets": targets, "rows": rows,
            "note": "central differences in each entry's own units (log or logit); pinned "
                    "entries evaluated across their bound; eta and head_extra perturbed on the "
                    "twin (fixed in the calibration)",
            "t_s": round(time.perf_counter() - t0, 1)}


def e10(S, H0, H1, Ech, obs_m, tm, elig, G, cl, P) -> dict:
    tw = S["tw"]
    steps = S["steps"]
    out: dict = {}
    # channel parts of the metered series
    mo = np.array([int(s[5:7]) for s in steps])
    mean_c = Ech.mean(axis=1, keepdims=True)
    clim = np.stack([Ech[:, mo == k].mean(axis=1) for k in range(1, 13)], axis=1)
    seas = clim[:, mo - 1] - mean_c
    anom = Ech - mean_c - seas
    parts = np.stack([np.broadcast_to(mean_c, Ech.shape), seas, anom], axis=-1)
    Es0 = tw.spread(S["E"])
    Es1 = tw.spread(S["E"] + Ech)
    X0 = L.tlm_response(S, H0, Es0, parts)
    X1 = L.tlm_response(S, H1, Es1, parts)
    r0_full = S["obs"] - P["A0"]
    r1_full = S["obs"] - P["A1"]
    tm2 = L.step_mask(S, "2014-01", L.SCORE_LAST)
    r0 = np.where(tm, r0_full, np.nan)
    r0_v2 = np.where(tm2, r0_full, np.nan)
    phi = lag1_phi(r0, elig)
    out["phi_A0_residual"] = phi
    out["channels"] = {
        "V1_12m_about_A0": channel_fit(X0, r0, elig, cl, phi=phi),
        "V1_12m_about_A1": channel_fit(X1, np.where(tm, r1_full, np.nan), elig, cl,
                                       names=("mean_minus_1", "seasonal_minus_1",
                                              "anomaly_minus_1")),
        "V2_24m_about_A0": channel_fit(X0, r0_v2, L.eligible(np.where(tm2, S["obs"], np.nan)),
                                       cl, phi=phi),
        "V3_12m_trend_about_A0": channel_fit(X0, r0, elig, cl, detrend=True),
    }
    # V4: consistent spin-up 2007-02..2012-01 from h0, recharge climatology
    Rfull = S["inp"].recharge_field.numpy()                       # (A, 132) 2012-01..2022-12
    m_all = np.array([int(m[5:7]) for m in S["E_months"]])
    Rclim = np.stack([Rfull[:, m_all == k].mean(axis=1) for k in range(1, 13)], axis=1)
    import pandas as pd

    sp_months = [str(d)[:7] for d in pd.date_range("2007-02-01", "2012-01-01", freq="MS")]
    sp_mo = np.array([int(m[5:7]) for m in sp_months])
    R_sp = Rclim[:, sp_mo - 1]
    E_sp_yl, mths, _ = L.c0_cells("preorigin", "model_design", counties=("yunlin",),
                                  first="2007-02", last="2012-01",
                                  grid_hash_expected=S["grid_hash"])
    E_sp_ch, _, _ = L.c0_cells("preorigin", "model_design", counties=("changhua", "chiayi"),
                               first="2007-02", last="2012-01",
                               grid_hash_expected=S["grid_hash"])
    assert list(mths) == sp_months
    Hs0 = L.rollout(S, E=E_sp_yl, R=R_sp)
    Hs1 = L.rollout(S, E=E_sp_yl, R=R_sp, E_extra=E_sp_ch)
    ic0, ic1 = Hs0[..., -1], Hs1[..., -1]
    H0s = L.rollout(S, h0=ic0)
    H1s = L.rollout(S, h0=ic1, E_extra=Ech)
    P0s, P1s = L.at_wells(S, H0s), L.at_wells(S, H1s)
    X0s = L.tlm_response(S, H0s, Es0, parts)
    r0s = np.where(tm, S["obs"] - P0s, np.nan)
    out["channels"]["V4_spinup_12m_about_A0s"] = channel_fit(X0s, r0s, elig, cl,
                                                             phi=lag1_phi(r0s, elig))
    out["spinup"] = {
        "months": [sp_months[0], sp_months[-1]], "recharge": "model recharge field, "
        "month-of-year climatology 2012-01..2022-12",
        "ic_change_vs_h0_layer1_changhua_cells_mean_m": {
            "A0": float((ic0 - S["h0"])[1][S["cell_county"] == L.CH].mean()),
            "A1": float((ic1 - S["h0"])[1][S["cell_county"] == L.CH].mean())},
        "ic_change_vs_h0_layer1_yunlin_cells_mean_m": {
            "A0": float((ic0 - S["h0"])[1][S["cell_county"] == L.YL].mean()),
            "A1": float((ic1 - S["h0"])[1][S["cell_county"] == L.YL].mean())}}
    # transient and trend
    arms = {"A0": P["A0"], "A1": P["A1"], "A0_spinup": P0s, "A1_spinup": P1s}
    out["transient"], out["trend"] = {}, {}
    for arm, pr in arms.items():
        resid = S["obs"] - pr
        tr, sl, tt = per_well_transient_trend(resid, steps, tm)
        out["transient"][arm] = {g: summarise_wellstat(tr, cl, G[g], "early (2012-02..2013-12)"
                                                       " minus late (2015-22) mean residual, m")
                                 for g in ("CH", "YL")}
        out["trend"][arm] = {g: {**summarise_wellstat(sl, cl, G[g] & np.isfinite(sl),
                                                      "residual slope 2013-22, m/yr"),
                                 "share_abs_t_gt_2_naive": float(np.mean(np.abs(
                                     tt[G[g] & np.isfinite(tt)]) > 2))}
                             for g in ("CH", "YL")}
    # interpretation helpers (declared rule)
    def covers0(d):
        return bool(d["p05"] <= 0 <= d["p95"])

    out["disappears"] = {
        kind: {g: {"A0_interval_excludes_0": not covers0(out[kind]["A0"][g]),
                   "A1_interval_covers_0": covers0(out[kind]["A1"][g]),
                   "disappears": (not covers0(out[kind]["A0"][g]))
                   and covers0(out[kind]["A1"][g])}
               for g in ("CH", "YL")} for kind in ("transient", "trend")}
    mean_neg = {v: bool(out["channels"][v]["mean"]["p95"] < 0)
                for v in ("V1_12m_about_A0", "V2_24m_about_A0", "V3_12m_trend_about_A0",
                          "V4_spinup_12m_about_A0s")}
    out["mean_channel_significantly_negative"] = mean_neg
    out["mean_channel_verdict"] = (
        "level channel misspecified (negative in every variant; plan E10)"
        if all(mean_neg.values()) else
        "not significantly negative in: " + ", ".join(k for k, v in mean_neg.items() if not v))
    out["spinup_T1"] = {g: {a: L.group_metrics(L.per_well_stats(obs_m, p), G[g])
                            for a, p in (("A0_spinup", P0s), ("A1_spinup", P1s))}
                        for g in ("CH", "YL")}
    L.log("T3 mean channel " + ", ".join(f"{k} {v['mean']['theta']:+.3f}"
                                         for k, v in out["channels"].items() if "mean" in v))
    return out


if __name__ == "__main__":
    raise SystemExit(main())

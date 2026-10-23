"""WP S5 repair round: after-the-fact diagnostics (report only; nothing feeds back into v2, a
gate, a threshold or a hyperparameter). Look logged first: ``s5r_looks.py repair_diagnostics``.

    tools/cpu_slot.sh python -I s5r_diag.py stageA     continuation of the stage-A optimiser
                                                       from the returned vertex (maxfev 300) and
                                                       the v2 product's sensitivity to it
    tools/cpu_slot.sh python -I s5r_diag.py t2buffer   T2 re-run with a 10 km buffer around each
                                                       held-out block (plan 4.3; design-exposed)
    tools/cpu_slot.sh python -I s5r_diag.py campaign   stage A with per-campaign leveling
                                                       offsets (does q leave its bound?)
    tools/cpu_slot.sh python -I s5r_diag.py variants   median increments of v2, v1 and the
                                                       GNSS-operator variants

Outputs: results/dataprep/reports/S5_evidence/repair/diag_<what>.json (+ tables).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5fit as F  # noqa: E402
import s5hyper as Hy  # noqa: E402
import s5model as M  # noqa: E402
import s5r_const as R  # noqa: E402
import s5r_core as RC  # noqa: E402
import s5r_vconst as V  # noqa: E402
import s5tests as Tt  # noqa: E402
from scipy import optimize  # noqa: E402

KNOTS_A = Hy.KNOTS_A


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def latest_product() -> Path:
    base = C.DATAPREP / C.PRODUCT
    vs = sorted((p for p in base.glob("v*") if (p / "MANIFEST.json").exists()),
                key=lambda p: int(p.name[1:]))
    return vs[-1]


# ============================================================================ stage A
def stage_a_continuation() -> dict:
    t0 = time.time()
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    data, _ = Hy.load("pre")
    keep = ~data["lev"]["rid"].isin(base["flagged"]["lev"]).to_numpy()
    pr = F.make_problem(data, KNOTS_A, ("lev",), masks={"lev": keep})
    h0 = dict(base["hyper"])
    m2_returned = F.fit(pr, h0).m2loglik
    tail = base["stages"]["A"]["refit"]["history_tail"]
    best_tail = min(tail, key=lambda r: r["m2ll"])
    hb = dict(h0)
    hb.update(best_tail["x"])
    m2_best_tail = F.fit(pr, hb).m2loglik
    h1, info = F.optimise(pr, h0, C.STAGES["A"], ("lev",), V.DIAG_STAGE_A_MAXFEV, log=log)
    out = {"note": "after the fact: Nelder-Mead continued from the registered stage-A vertex "
                   "(same objective, problem and bounds); report only",
           "returned_vertex": {k: h0[k] for k in C.STAGES["A"]}, "m2ll_returned": m2_returned,
           "best_evaluated_in_registered_run": {"x": best_tail["x"], "m2ll": m2_best_tail},
           "continued": {k: h1[k] for k in C.STAGES["A"]}, "continuation": info,
           "m2ll_drop": m2_returned - info["m2ll"]}
    log(f"stage A continued: m2ll {m2_returned:.3f} -> {info['m2ll']:.3f}; "
        f"{json.dumps({k: round(h1[k], 3) for k in C.STAGES['A']})}")
    # sensitivity of the v2 product to the continued stage-A values (stages B-D unchanged)
    d = latest_product()
    hp = json.loads((d / "hyperparameters.json").read_text())
    steps = pd.read_csv(d / "leveling_steps.csv")
    dfull, _ = Hy.load("full")
    breaks = RC.gnss_breaks(RC.read_daily("full"))
    dfull["gnss"]["seg"] = RC.gnss_segments(dfull["gnss"], breaks)
    dfull["lev"]["seg"] = RC.lev_segments(dfull["lev"], steps)
    m = Hy.masks_from(dfull, hp["screened_rids"])
    hc = dict(hp["hyper"])
    hc.update({k: h1[k] for k in C.STAGES["A"]})
    prf = RC.make_problem_v2(dfull, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=hp["fdict"])
    post = RC.fit_v2(prf, hc)
    F.set_ell(prf, hc["ell_km"])
    Phi = prf.lat.weights(dfull["grid"]["cell_xy_km"])
    z = np.load(d / "fused_annual.npz", allow_pickle=False)
    years = z["years"]
    Zm = post.theta[:prf.L].reshape(prf.K, prf.m)
    inc = np.stack([Phi @ ((prf.tb.knot_row(i + 1) - prf.tb.knot_row(i)) @ Zm)
                    + post.theta[prf.L] for i in range(years.size)])
    sel = (years >= 2020) & (years <= 2025)
    dlt = inc[sel] - z["inc"][sel]
    out["product_sensitivity"] = {
        "product": f"{C.PRODUCT}/{d.name}",
        "rms_inc_2020_2025": rmse(dlt), "max_abs_inc_2020_2025": float(np.max(np.abs(dlt))),
        "rms_by_year": {int(y): rmse(dlt[j]) for j, y in enumerate(years[sel])},
        "median_inc_sd_product": {int(y): float(np.median(z["inc_sd"][i]))
                                  for i, y in enumerate(years) if 2020 <= y <= 2025}}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ============================================================================ T2 with a buffer
def t2_buffer() -> dict:
    """Registered T2 design (round-1 operator, pre-origin, hyper_pre.json) with every
    benchmark within T2B_BUFFER_KM of a held-out benchmark also left out of the fit; IDW of
    neighbours outside block and buffer within T2B_IDW_RADIUS_KM. Design-exposed: the T2
    windows and blocks were scored on 2026-10-14."""
    t0 = time.time()
    base, data, m = Tt.base_setup()
    h, fdict = base["hyper"], base["fdict"]
    lev = data["lev"][m["lev"]]
    cl = Tt.benchmark_clusters(lev)
    pr = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    c_all = F.total_contribution(pr, h)
    bxy = lev.groupby("sid")[["x_km", "y_km"]].first()
    rates = {}
    for sid, g in lev.groupby("sid"):
        if len(g) >= C.T1_BASE_MIN_SURVEYS and g.t.max() - g.t.min() >= C.T1_BASE_MIN_SPAN_YR:
            rates[sid] = (Tt.ls_rate(g.t, g.y), g[["x_km", "y_km"]].to_numpy()[0])
    rs = np.array(list(rates.keys()))
    rxy = np.array([v[1] for v in rates.values()])
    rv = np.array([v[0] for v in rates.values()])
    rows = []
    blocks = sorted(set(cl.values()))
    for k, b in enumerate(blocks):
        sids = [s for s, c in cl.items() if c == b]
        xyb = bxy.loc[sids].to_numpy()
        dmin = M.dist(bxy.to_numpy(), xyb).min(1)
        out_ids = bxy.index[dmin <= V.T2B_BUFFER_KM].tolist()
        cb = F.lev_subset_contribution(pr, h, out_ids)
        post = F.fit(pr, h, contrib=F.subtract(c_all, cb))
        dr = M.dist(rxy, xyb).min(1)
        nb_ok = ~np.isin(rs, out_ids)
        for sid in sids:
            g = lev[lev.sid == sid].sort_values("t")
            if len(g) < 2:
                continue
            xy = g[["x_km", "y_km"]].to_numpy()[0]
            t = g.t.to_numpy()
            G = F.field_rows(pr, xy, t[1:]) - F.field_rows(pr, xy, t[:-1])
            mu = G @ post.theta
            dt = np.diff(t)
            var = post.var_of(G) + dt ** 2 * h["sig_rb"] ** 2 + 2 * h["sig_lev"] ** 2
            d = np.sqrt(((rxy - xy) ** 2).sum(1))
            nb = nb_ok & (d <= V.T2B_IDW_RADIUS_KM)
            if nb.sum() >= C.T2_IDW_MIN_NEIGHBOURS:
                w = 1.0 / np.maximum(d[nb], 0.1) ** C.T2_IDW_POWER
                idw = float(np.sum(w * rv[nb]) / np.sum(w))
            else:
                idw = np.nan
            obs = np.diff(g.y.to_numpy())
            for j in range(len(dt)):
                rows.append({"sid": sid, "cluster": int(b), "t0": float(t[j]),
                             "t1": float(t[j + 1]), "pred_fused": float(mu[j]),
                             "sd_fused": float(np.sqrt(var[j])),
                             "pred_idw": idw * float(dt[j]) if np.isfinite(idw) else np.nan,
                             "obs": float(obs[j]), "n_left_out": len(out_ids),
                             "nearest_kept_km": float(dr[nb_ok].min()) if nb_ok.any()
                             else np.nan})
        if (k + 1) % 10 == 0:
            log(f"T2-buffer block {k + 1}/{len(blocks)} ({time.time() - t0:.0f} s)")
    T = pd.DataFrame(rows)
    T["err_fused"] = T.obs - T.pred_fused
    T["err_idw"] = T.obs - T.pred_idw
    T["hit_fused"] = np.abs(T.err_fused) <= Tt.Z90 * T.sd_fused
    B = T[np.isfinite(T.pred_idw)]
    reg = json.loads((C.EVIDENCE / "t2.json").read_text())
    out = {"note": "after the fact, design-exposed (registered T2 blocks and windows), report "
                   "only: T2 with a 10 km buffer (plan 4.3) and IDW of neighbours beyond the "
                   "buffer within 20 km",
           "n_blocks": len(blocks), "n_targets": int(len(T)),
           "median_benchmarks_left_out": float(T.groupby("cluster").n_left_out.first().median()),
           "coverage_fused": Tt.coverage(T.hit_fused, T.cluster),
           "rmse_fused_all": rmse(T.err_fused), "n_with_idw": int(len(B)),
           "rmse_fused": rmse(B.err_fused), "rmse_idw": rmse(B.err_idw),
           "ratio_fused_idw": rmse(B.err_fused) / max(rmse(B.err_idw), 1e-12),
           "boot_ratio": Tt.boot_ratio(B.err_fused, B.err_idw, B.cluster),
           "rms_z": float(np.sqrt(np.mean((T.err_fused / T.sd_fused) ** 2))),
           "registered_no_buffer": {"rmse_fused": reg["rmse_fused"], "rmse_idw": reg["rmse_idw"],
                                    "ratio": reg["ratio_fused_idw"],
                                    "coverage": reg["coverage_fused"]["coverage"]},
           "seconds": round(time.time() - t0, 1)}
    T.to_parquet(R.EVIDENCE_R / "diag_t2buffer_targets.parquet", index=False)
    return out


# ============================================================================ campaigns
def county_of_cells() -> dict:
    units = json.loads(C.UNITS_V3.read_text())["units"]
    out = {}
    for u in units:
        if u["id"] in ("dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3"):
            for c in u["cells"]:
                out[int(c)] = "CH"
        elif u["id"] in ("dt.tercile.west", "dt.tercile.mid", "dt.tercile.east"):
            for c in u["cells"]:
                out[int(c)] = "YL"
    return out


def campaign_stage_a() -> dict:
    """Stage A (leveling only, pre-origin screening) with a random offset per survey campaign
    (calendar month x county of the benchmark's cell), prior sd sig_camp (a 7th
    hyperparameter); started from the registered stage-A values and sig_camp 3 mm."""
    t0 = time.time()
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    data, _ = Hy.load("pre")
    keep = ~data["lev"]["rid"].isin(base["flagged"]["lev"]).to_numpy()
    cc = county_of_cells()
    lev = data["lev"]
    camp = (lev["date"].str[:7] + ":" + lev["cell"].map(lambda c: cc.get(int(c), "OT")))
    data["lev"] = lev.assign(camp=camp)
    pr = F.make_problem(data, KNOTS_A, ("lev",), masks={"lev": keep})
    camps = sorted(pr.lev.camp.unique())
    n0 = len(pr.gnames)
    for c in camps:
        pr.gnames.append(f"lcamp:{c}")
        pr.gindex[f"lcamp:{c}"] = len(pr.gnames) - 1
    for s in pr._lev_sites:
        cs = pr.lev.loc[s["rows"], "camp"].to_numpy()
        s["camp_idx"] = np.array([pr.gindex[f"lcamp:{c}"] for c in cs])

    def contrib(h):
        F.set_ell(pr, h["ell_km"])
        asm = M.Assembler(pr.K, pr.m, pr.ng)
        for s in pr._lev_sites:
            W, ld, ne = M.site_local(s["t"], np.zeros(s["t"].size, int), h["sig_lev"],
                                     h["sig_rb"])
            uc, inv = np.unique(s["camp_idx"], return_inverse=True)
            G = np.zeros((s["t"].size, 1 + uc.size))
            G[:, 0] = s["t"] - C.T_REF
            G[np.arange(s["t"].size), 1 + inv] = 1.0
            asm.add_site(s["phi"], s["H"], W, s["y"], gidx=np.r_[0, uc], G=G, logdet=ld,
                         n_eff=ne)
        return asm.result()

    def m2ll(h):
        c = contrib(h)
        J = pr.tb.prior_precision(h["sig_v0"], h["q"], h["sig_s"])
        gp = np.full(pr.ng, 1.0 / C.VAGUE_SD ** 2)
        gp[0] = 1.0 / C.MU_V_SD ** 2
        gp[n0:] = 1.0 / h["sig_camp"] ** 2
        return M.solve(c, J, pr.lat, gp).m2loglik

    names = tuple(C.STAGES["A"]) + ("sig_camp",)
    bounds = dict(C.HYPER_BOUNDS)
    bounds["sig_camp"] = (0.05, 30.0)
    lo = np.log([bounds[n][0] for n in names])
    hi = np.log([bounds[n][1] for n in names])
    h = dict(base["hyper"])
    h["sig_camp"] = V.CAMPAIGN_SD_START
    hist = []

    def obj(x):
        xc = np.clip(x, lo, hi)
        hh = dict(h)
        hh.update({n: float(np.exp(v)) for n, v in zip(names, xc, strict=True)})
        try:
            v = m2ll(hh) + 1e3 * float(np.sum((x - xc) ** 2))
        except np.linalg.LinAlgError:
            v = 1e18
        hist.append({"x": {n: hh[n] for n in names}, "m2ll": float(v)})
        if len(hist) % 20 == 0:
            log(f"  campaign eval {len(hist)} m2ll {v:.3f} best "
                f"{min(r['m2ll'] for r in hist):.3f}")
        return v

    x0 = np.log([h[n] for n in names])
    S = np.tile(x0, (len(x0) + 1, 1))
    for i in range(len(x0)):
        S[i + 1, i] += 0.35
    res = optimize.minimize(obj, x0, method="Nelder-Mead",
                            options={"maxfev": V.DIAG_STAGE_A_MAXFEV, "xatol": 1e-3,
                                     "fatol": 0.05, "initial_simplex": S})
    best = min(hist, key=lambda r: r["m2ll"])
    # the same problem without campaign offsets at the registered values (sig_camp -> 0.05)
    h_nc = dict(h)
    h_nc["sig_camp"] = 0.05
    out = {"note": "after the fact, report only: stage A with per-campaign leveling offsets "
                   "(campaign = survey month x county of the benchmark cell)",
           "n_campaigns": len(camps), "campaigns": camps,
           "registered_stage_a": {k: base["hyper"][k] for k in C.STAGES["A"]},
           "m2ll_registered_values_sig_camp_0.05": m2ll(h_nc), "best": best,
           "nfev": int(res.nfev), "converged": bool(res.success),
           "q_at_upper_bound": bool(best["x"]["q"] > 0.999 * C.HYPER_BOUNDS["q"][1]),
           "m2ll_registered_stage_a_refit": base["stages"]["A"]["refit"]["m2ll"],
           "seconds": round(time.time() - t0, 1)}
    log(f"campaign stage A: best m2ll {best['m2ll']:.3f} "
        f"{json.dumps({k: round(v, 3) for k, v in best['x'].items()})}")
    return out


# ============================================================================ variants
def variants() -> dict:
    """Fan-wide and county median annual increments 2020-2025 of v2 and of the GNSS-operator
    variants of sensitivity.json (v1 operator; SJES breaks only), same screening as v2."""
    t0 = time.time()
    d = latest_product()
    hp = json.loads((d / "hyperparameters.json").read_text())
    steps = pd.read_csv(d / "leveling_steps.csv")
    data, _ = Hy.load("full")
    breaks = RC.gnss_breaks(RC.read_daily("full"))
    data["lev"]["seg"] = RC.lev_segments(data["lev"], steps)
    m = Hy.masks_from(data, hp["screened_rids"])
    h, fdict = hp["hyper"], hp["fdict"]
    z = np.load(d / "fused_annual.npz", allow_pickle=False)
    z1 = np.load(C.DATAPREP / C.PRODUCT / "v1" / "fused_annual.npz", allow_pickle=False)
    years = z["years"]
    cc = county_of_cells()
    grp = np.array([cc.get(c, "OT") for c in range(data["grid"]["A"])])
    out = {"note": "after the fact, report only", "product": f"{C.PRODUCT}/{d.name}"}

    def summ(inc):
        sel = (years >= 2020) & (years <= 2025)
        return {g: {int(y): float(np.median(inc[i][grp == g] if g != "all" else inc[i]))
                    for i, y in zip(np.flatnonzero(sel), years[sel], strict=True)}
                for g in ("all", "CH", "YL")}

    out["v2"] = summ(z["inc"])
    out["v1_product"] = summ(z1["inc"])
    for name, br in (("v1_operator", None), ("sjes_breaks_only", {V.SJES: breaks[V.SJES]})):
        if br is None:
            pr = F.make_problem(data, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=fdict)
            post = F.fit(pr, h)
        else:
            data["gnss"]["seg"] = RC.gnss_segments(data["gnss"], br)
            pr = RC.make_problem_v2(data, C.KNOTS_FULL, F.FAMILIES, masks=m, fdict=fdict)
            post = RC.fit_v2(pr, h)
        F.set_ell(pr, h["ell_km"])
        Phi = pr.lat.weights(data["grid"]["cell_xy_km"])
        Zm = post.theta[:pr.L].reshape(pr.K, pr.m)
        inc = np.stack([Phi @ ((pr.tb.knot_row(i + 1) - pr.tb.knot_row(i)) @ Zm)
                        + post.theta[pr.L] for i in range(years.size)])
        out[name] = summ(inc)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stageA", "t2buffer", "campaign", "variants"])
    a = ap.parse_args(argv)
    R.EVIDENCE_R.mkdir(parents=True, exist_ok=True)
    out = R.EVIDENCE_R / f"diag_{a.what}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    res = {"stageA": stage_a_continuation, "t2buffer": t2_buffer,
           "campaign": campaign_stage_a, "variants": variants}[a.what]()
    F.dump(res, out)
    log(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

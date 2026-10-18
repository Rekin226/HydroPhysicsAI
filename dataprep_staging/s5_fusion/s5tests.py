"""WP S5: the registered tests of ``S5.subsidence_fusion.v1``.

    tools/cpu_slot.sh python -I s5tests.py t1     T1 (gate): leveling 2020-21 withheld, the
                                                  mirror of the real post-2021 gap
    tools/cpu_slot.sh python -I s5tests.py t2     T2 (gate): leave one 10 km leveling block out
    tools/cpu_slot.sh python -I s5tests.py t3     T3 (report only): leave one instrument family
                                                  out, pre-origin
    tools/cpu_slot.sh python -I s5tests.py r2     R2 (report only, U7: never a gate): post-2022
                                                  family hold-out on the full data

Each test calls ``prereg3.assert_can_score`` after its fits and predictions and immediately
before comparing any prediction with a withheld observation. T1-T3 use pre-origin data only
(hyperparameters from S5_evidence/hyper_pre.json, T1 stage A from hyper_t1.json).
Outputs: results/dataprep/reports/S5_evidence/<test>.json and <test>_targets.parquet.
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
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

Z90 = 1.6448536269514722
TEST_IDS = {"t1": "S5.T1_leveling_2020_2021_withheld", "t2": "S5.T2_leveling_block_out",
            "t3": "S5.T3_family_out_preorigin", "r2": "S5.R2_post2022_family_holdout"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def score_gate(key: str) -> dict:
    from harness import prereg3
    r = prereg3.assert_can_score(C.PREREG_ID, C.WP, test_id=TEST_IDS[key])
    return {"prereg_sha256": r["sha256"], "post_registration_looks":
            len(r["post_registration_looks"]),
            "unverifiable_prior_looks": len(r["unverifiable_prior_looks"])}


# ============================================================================ helpers
def benchmark_clusters(lev: pd.DataFrame) -> dict:
    """sid -> block label: complete-linkage clusters of benchmarks, diameter <= 10 km."""
    b = lev.groupby("sid")[["x_km", "y_km"]].first().sort_index()
    Z = linkage(b.to_numpy(), method="complete")
    lab = fcluster(Z, t=C.T2_CLUSTER_KM, criterion="distance")
    return dict(zip(b.index, lab.astype(int), strict=True))


def coverage(hits, clusters) -> dict:
    from harness import coverage3
    hits = np.asarray(hits, bool)
    if hits.size == 0:
        return {"n": 0, "pass": False, "note": "no targets"}
    _, cl = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return coverage3.coverage_gate(hits.astype(int), cl)


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def boot_ratio(e_a, e_b, clusters, n=C.BOOT_N, seed=C.SEED) -> dict:
    """Cluster bootstrap of RMSE(e_a) / RMSE(e_b)."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    if e_a.size == 0:
        return {}
    cl = np.asarray(clusters).astype(str)
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(u, size=u.size, replace=True)])
        out[i] = rmse(e_a[pick]) / max(rmse(e_b[pick]), 1e-12)
    return {"p05": float(np.quantile(out, 0.05)), "p50": float(np.quantile(out, 0.5)),
            "p95": float(np.quantile(out, 0.95)), "p_ge_1": float(np.mean(out >= 1.0)),
            "n_boot": n, "seed": seed, "k_clusters": int(u.size)}


def ls_rate(t, y) -> float:
    t = np.asarray(t, float)
    return float(np.polyfit(t - t.mean(), np.asarray(y, float), 1)[0])


def pred_time_heldout(pr, post, h, xy, t_tr, y_tr, t_new):
    """Predictive of a benchmark's new surveys given its own training surveys (exact: local
    offset and random velocity conditioned on the training surveys and theta)."""
    t_tr, y_tr, t_new = map(lambda a: np.asarray(a, float), (t_tr, y_tr, t_new))
    G_tr = F.field_rows(pr, xy, t_tr)
    G_nw = F.field_rows(pr, xy, t_new)
    tb = t_tr.mean()
    L_tr = np.column_stack([np.ones(t_tr.size), t_tr - tb])
    L_nw = np.column_stack([np.ones(t_new.size), t_new - tb])
    s2 = h["sig_lev"] ** 2
    Lam = L_tr.T @ L_tr / s2
    Lam[1, 1] += 1.0 / h["sig_rb"] ** 2
    Li = np.linalg.inv(Lam)
    Mm = Li @ L_tr.T / s2
    g = G_nw - L_nw @ Mm @ G_tr
    mean = g @ post.theta + L_nw @ Mm @ y_tr
    var = post.var_of(g) + np.einsum("ij,jk,ik->i", L_nw, Li, L_nw) + s2
    return mean, np.sqrt(var)


def field_increment_var(pr, post, xy, t0, t1) -> np.ndarray:
    """Posterior variance of the field increment u(x, t1) - u(x, t0)."""
    t1 = np.atleast_1d(np.asarray(t1, float))
    t0 = np.broadcast_to(np.asarray(t0, float), t1.shape).copy()
    g = F.field_rows(pr, xy, t1) - F.field_rows(pr, xy, t0)
    return post.var_of(g)


def base_setup():
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    data, _ = Hy.load("pre")
    m = Hy.masks_from(data, base["flagged"])
    return base, data, m


# ============================================================================ T1
def t1() -> dict:
    t0 = time.time()
    base, data, m = base_setup()
    h1 = json.loads((C.EVIDENCE / "hyper_t1.json").read_text())["hyper"]
    lev = data["lev"]
    train = m["lev"] & (lev["date"].to_numpy() <= C.T1_BASE_WIN[1])
    held = m["lev"] & (lev["date"].to_numpy() >= C.T1_WITHHELD[0]) & \
        (lev["date"].to_numpy() <= C.T1_WITHHELD[1])
    masks = dict(m)
    masks["lev"] = train
    pr_c = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=masks)
    post_c = F.fit(pr_c, h1)
    fd = F.estimate_f(pr_c, post_c, data["cmp"][m["cmp"]])
    fdict = {k: v["f"] for k, v in fd.items()}
    del post_c
    pr_f = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=masks, fdict=fdict)
    post_f = F.fit(pr_f, h1)
    pr_l = F.make_problem(data, C.KNOTS_PRE, ("lev",), masks={"lev": train})
    post_l = F.fit(pr_l, h1)
    log(f"T1 fits done ({time.time() - t0:.0f} s)")
    cl = benchmark_clusters(lev[m["lev"]])
    rows = []
    for sid in sorted(lev.loc[held, "sid"].unique()):
        tr = lev[train & (lev.sid == sid).to_numpy()]
        nw = lev[held & (lev.sid == sid).to_numpy()]
        if len(tr) < C.T1_BASE_MIN_SURVEYS or tr.t.max() - tr.t.min() < C.T1_BASE_MIN_SPAN_YR:
            continue
        xy = tr[["x_km", "y_km"]].to_numpy()[0]
        ta, ya = float(tr.t.iloc[-1]), float(tr.y.iloc[-1])
        mf, sf = pred_time_heldout(pr_f, post_f, h1, xy, tr.t, tr.y, nw.t)
        ml, sl = pred_time_heldout(pr_l, post_l, h1, xy, tr.t, tr.y, nw.t)
        vf = field_increment_var(pr_f, post_f, xy, ta, nw.t.to_numpy())
        vl = field_increment_var(pr_l, post_l, xy, ta, nw.t.to_numpy())
        rate = ls_rate(tr.t, tr.y)
        for j in range(len(nw)):
            rows.append({"sid": sid, "cluster": cl.get(sid, -1), "t_a": ta,
                         "t_new": float(nw.t.iloc[j]), "pred_fused": float(mf[j] - ya),
                         "sd_fused": float(sf[j]), "pred_levonly": float(ml[j] - ya),
                         "sd_levonly": float(sl[j]),
                         "pred_baseline": rate * (float(nw.t.iloc[j]) - ta),
                         "var_field_fused": float(vf[j]), "var_field_levonly": float(vl[j]),
                         "vr": float(1.0 - vf[j] / vl[j]),
                         "_obs": float(nw.y.iloc[j] - ya)})
    T = pd.DataFrame(rows)
    log(f"T1 {len(T)} targets at {T.sid.nunique() if len(T) else 0} benchmarks")
    reg = score_gate("t1")
    # ---------------- score (first comparison with withheld observations)
    T["obs"] = T.pop("_obs")
    T["err_fused"] = T.obs - T.pred_fused
    T["err_levonly"] = T.obs - T.pred_levonly
    T["err_baseline"] = T.obs - T.pred_baseline
    T["hit_fused"] = np.abs(T.err_fused) <= Z90 * T.sd_fused
    T["hit_levonly"] = np.abs(T.err_levonly) <= Z90 * T.sd_levonly
    S = T[T.vr >= C.VR_MIN]
    res = {"test_id": TEST_IDS["t1"], "registration": reg, "hyper_t1": h1, "f_t1": fd,
           "n_targets_all": int(len(T)), "n_benchmarks_all": int(T.sid.nunique()),
           "n_targets_scored": int(len(S)), "n_benchmarks_scored": int(S.sid.nunique()),
           "vr_quantiles": {q: float(T.vr.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    if res["n_benchmarks_scored"] >= C.T1_MIN_SCORED:
        res["scored"] = {"rmse_fused": rmse(S.err_fused), "rmse_baseline": rmse(S.err_baseline),
                         "rmse_levonly": rmse(S.err_levonly),
                         "ratio_fused_baseline": rmse(S.err_fused) / rmse(S.err_baseline),
                         "boot_ratio": boot_ratio(S.err_fused, S.err_baseline, S.cluster),
                         "bias_fused": float(S.err_fused.mean()),
                         "bias_baseline": float(S.err_baseline.mean()),
                         "coverage_fused": coverage(S.hit_fused, S.cluster),
                         "coverage_levonly": coverage(S.hit_levonly, S.cluster)}
        sc = res["scored"]
        res["criteria"] = {"evaluable": True,
                           "skill": bool(sc["ratio_fused_baseline"] < 1.0),
                           "coverage": bool(sc["coverage_fused"]["pass"])}
    else:
        res["criteria"] = {"evaluable": False, "skill": False, "coverage": False}
    res["pass"] = bool(all(res["criteria"].values()))
    res["all_targets"] = {"rmse_fused": rmse(T.err_fused), "rmse_baseline": rmse(T.err_baseline),
                          "rmse_levonly": rmse(T.err_levonly),
                          "coverage_fused": coverage(T.hit_fused, T.cluster)}
    res["seconds"] = round(time.time() - t0, 1)
    T.to_parquet(C.EVIDENCE / "t1_targets.parquet", index=False)
    F.dump(res, C.EVIDENCE / "t1.json")
    log(f"T1 pass {res['pass']} {json.dumps(res['criteria'])}")
    return res


# ============================================================================ T2
def t2() -> dict:
    t0 = time.time()
    base, data, m = base_setup()
    h, fdict = base["hyper"], base["fdict"]
    lev = data["lev"][m["lev"]]
    cl = benchmark_clusters(lev)
    pr = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    c_all = F.total_contribution(pr, h)
    # IDW baseline rates (every screened benchmark with >= 3 surveys spanning >= 3 yr)
    rates = {}
    for sid, g in lev.groupby("sid"):
        if len(g) >= C.T1_BASE_MIN_SURVEYS and g.t.max() - g.t.min() >= C.T1_BASE_MIN_SPAN_YR:
            rates[sid] = (ls_rate(g.t, g.y), g[["x_km", "y_km"]].to_numpy()[0])
    rs = np.array(list(rates.keys()))
    rxy = np.array([v[1] for v in rates.values()])
    rv = np.array([v[0] for v in rates.values()])
    rcl = np.array([cl[s] for s in rs])
    rows = []
    blocks = sorted(set(cl.values()))
    for k, b in enumerate(blocks):
        sids = [s for s, c in cl.items() if c == b]
        cb = F.lev_subset_contribution(pr, h, sids)
        post = F.fit(pr, h, contrib=F.subtract(c_all, cb))
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
            nb = (rcl != b) & (d <= C.T2_IDW_RADIUS_KM)
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
                             "_obs": float(obs[j])})
        if (k + 1) % 10 == 0:
            log(f"T2 block {k + 1}/{len(blocks)} ({time.time() - t0:.0f} s)")
    T = pd.DataFrame(rows)
    reg = score_gate("t2")
    # ---------------- score
    T["obs"] = T.pop("_obs")
    T["err_fused"] = T.obs - T.pred_fused
    T["err_idw"] = T.obs - T.pred_idw
    T["hit_fused"] = np.abs(T.err_fused) <= Z90 * T.sd_fused
    B = T[np.isfinite(T.pred_idw)]
    res = {"test_id": TEST_IDS["t2"], "registration": reg, "n_blocks": len(blocks),
           "n_targets": int(len(T)), "n_benchmarks": int(T.sid.nunique()),
           "n_with_baseline": int(len(B)),
           "coverage_fused": coverage(T.hit_fused, T.cluster),
           "rmse_fused": rmse(B.err_fused), "rmse_idw": rmse(B.err_idw),
           "ratio_fused_idw": rmse(B.err_fused) / rmse(B.err_idw),
           "boot_ratio": boot_ratio(B.err_fused, B.err_idw, B.cluster),
           "rmse_fused_all": rmse(T.err_fused), "bias_fused": float(T.err_fused.mean()),
           "median_sd_fused": float(T.sd_fused.median())}
    res["criteria"] = {"coverage": bool(res["coverage_fused"]["pass"]),
                       "skill": bool(res["ratio_fused_idw"] < 1.0)}
    res["pass"] = bool(all(res["criteria"].values()))
    res["seconds"] = round(time.time() - t0, 1)
    T.to_parquet(C.EVIDENCE / "t2_targets.parquet", index=False)
    F.dump(res, C.EVIDENCE / "t2.json")
    log(f"T2 pass {res['pass']} {json.dumps(res['criteria'])}")
    return res


# ============================================================================ T3
def _detrend_proj(X: np.ndarray) -> np.ndarray:
    return np.eye(len(X)) - X @ np.linalg.pinv(X)


def t3() -> dict:
    t0 = time.time()
    base, data, m = base_setup()
    h, fdict = base["hyper"], base["fdict"]
    out = {"test_id": TEST_IDS["t3"]}
    pending = {}
    # ---- GNSS out
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "insar", "cmp"), masks=m, fdict=fdict)
    post = F.fit(pr, h)
    gn = data["gnss"][m["gnss"]]
    rows = []
    for (fam, st), g in gn.groupby(["family", "station"]):
        t = g.t.to_numpy()
        X = np.column_stack([np.ones(t.size), t - t.mean()])
        P = _detrend_proj(X)
        G = F.field_rows(pr, g[["x_km", "y_km"]].to_numpy()[0], t)
        Sg = post.cov_of(G) + np.diag(h["sig_cm"] ** 2 + h["sig_gnss"] ** 2 + g.sd.to_numpy() ** 2)
        zhat = P @ (G @ post.theta)
        sd = np.sqrt(np.clip(np.diag(P @ Sg @ P.T), 1e-12, None))
        rows.append(pd.DataFrame({"family": fam, "station": st, "t": t, "zhat": zhat, "sd": sd,
                                  "_y": g.y.to_numpy(), "_P": [P] * t.size}))
    pending["gnss"] = rows
    # ---- rings and piles out
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=m)
    post = F.fit(pr, h)
    cm = data["cmp"][m["cmp"]]
    rows = []
    for sen, g in cm.groupby("sensor"):
        t = g.t.to_numpy()
        segs, inv = np.unique(g.seg.to_numpy(), return_inverse=True)
        X = np.zeros((t.size, 2 * segs.size))
        for k in range(segs.size):
            sel = inv == k
            X[sel, k] = 1.0
            X[sel, segs.size + k] = t[sel] - t[sel].mean()
        P = _detrend_proj(X)
        f = float(fdict.get(sen, 1.0))
        G = f * F.field_rows(pr, g[["x_km", "y_km"]].to_numpy()[0], t)
        Sg = post.cov_of(G) + h["sig_cmp"] ** 2 * np.eye(t.size)
        zhat = P @ (G @ post.theta)
        sd = np.sqrt(np.clip(np.diag(P @ Sg @ P.T), 1e-12, None))
        rows.append(pd.DataFrame({"sensor": sen, "t": t, "zhat": zhat, "sd": sd,
                                  "_y": g.y.to_numpy(), "_P": [P] * t.size}))
    pending["cmp"] = rows
    # ---- InSAR out
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "cmp"), masks=m, fdict=fdict)
    post = F.fit(pr, h)
    prl = F.make_problem(data, C.KNOTS_PRE, ("lev",), masks=m)
    postl = F.fit(prl, h)
    F.set_ell(pr, h["ell_km"])
    Pc = pr.lat.weights(pr.cell_xy)
    ins = data["insar"]
    series = {}
    for gr in ins["groups"]:
        for u, y in zip(gr["unit"], gr["y"], strict=True):
            series.setdefault(int(u), []).append((gr["t"], y))
    urows = []
    for u, s in series.items():
        if len(s) < 6:
            continue
        t = np.array([a for a, _ in s])
        yv = np.array([b for _, b in s])
        phi = Pc[ins["unit_cells"][u]].mean(0)
        H = pr.tb.rows(t)
        Zf = post.theta[:pr.L].reshape(pr.K, pr.m)
        Zl = postl.theta[:prl.L].reshape(prl.K, prl.m)
        uf = H @ (Zf @ phi) + post.theta[pr.L] * (t - C.T_REF)
        ul = H @ (Zl @ phi) + postl.theta[prl.L] * (t - C.T_REF)
        urows.append({"unit": u, "seg": int(ins["units"].loc[u, "seg"]),
                      "rate_fused": ls_rate(t, uf), "rate_levonly": ls_rate(t, ul),
                      "_rate_obs": ls_rate(t, yv)})
    U = pd.DataFrame(urows)
    reg = score_gate("t3")
    out["registration"] = reg
    # ---------------- score
    for fam in ("gnss", "cmp"):
        R = pd.concat(pending[fam], ignore_index=True)
        z = []
        for _key, g in R.groupby(["family", "station"] if fam == "gnss" else ["sensor"]):
            P = g["_P"].iloc[0]
            z.append(pd.Series(P @ g["_y"].to_numpy(), index=g.index))
        R["z"] = pd.concat(z)
        R = R.drop(columns=["_P", "_y"])
        R["err"] = R.z - R.zhat
        R["hit"] = np.abs(R.err) <= Z90 * R.sd
        clu = R.station if fam == "gnss" else R.sensor
        out[fam] = {"n": int(len(R)), "sites": int(clu.nunique()), "rmse_fused": rmse(R.err),
                    "rmse_zero_baseline": rmse(R.z), "ratio": rmse(R.err) / rmse(R.z),
                    "coverage": coverage(R.hit, clu)}
        if fam == "gnss":
            out[fam]["by_family"] = {f: {"rmse_fused": rmse(g.err), "rmse_zero": rmse(g.z),
                                         "n": int(len(g))} for f, g in R.groupby("family")}
        R.to_parquet(C.EVIDENCE / f"t3_{fam}_targets.parquet", index=False)
    U["rate_obs"] = U.pop("_rate_obs")
    for col in ("rate_obs", "rate_fused", "rate_levonly"):
        U[col + "_rel"] = U[col] - U.groupby("seg")[col].transform("median")
    out["insar"] = {"units": int(len(U)),
                    "r_fused": float(np.corrcoef(U.rate_obs_rel, U.rate_fused_rel)[0, 1]),
                    "r_levonly": float(np.corrcoef(U.rate_obs_rel, U.rate_levonly_rel)[0, 1]),
                    "rmse_fused": rmse(U.rate_obs_rel - U.rate_fused_rel),
                    "rmse_levonly": rmse(U.rate_obs_rel - U.rate_levonly_rel),
                    "rmse_zero": rmse(U.rate_obs_rel), "note": "rates in mm/yr relative to the "
                    "segment median (uniform part removed; tilt not removed)"}
    U.to_parquet(C.EVIDENCE / "t3_insar_units.parquet", index=False)
    out["seconds"] = round(time.time() - t0, 1)
    out["pass"] = None
    F.dump(out, C.EVIDENCE / "t3.json")
    log("T3 done")
    return out


# ============================================================================ R2 (after the fact)
def _withhold(data: dict, fam: str) -> dict:
    t23 = 2023.0
    ex = {}
    if fam in ("gnss_wra", "gnss_ngl"):
        f = fam.split("_")[1]
        g = data["gnss"]
        ex["gnss"] = ~((g.family == f) & (g.t >= t23)).to_numpy()
    elif fam in ("piles", "rings"):
        k = "pile" if fam == "piles" else "ring"
        c = data["cmp"]
        ex["cmp"] = ~((c.kind == k) & (c.t > t23 + 1e-6)).to_numpy()
    elif fam == "insar":
        ex["insar"] = np.array([gr["t"] < t23 for gr in data["insar"]["groups"]])
    return ex


def r2() -> dict:
    t0 = time.time()
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    h, fdict = base["hyper"], base["fdict"]
    data, _ = Hy.load("full")
    flagged = {"lev": base["flagged"]["lev"]}
    m0 = Hy.masks_from(data, flagged)
    pr = F.make_problem(data, C.KNOTS_FULL, F.FAMILIES, masks=m0, fdict=fdict)
    post = F.fit(pr, h)
    flagged["gnss"] = F.outliers(pr, post, h, "gnss")
    flagged["cmp"] = F.outliers(pr, post, h, "cmp")
    del post
    m = Hy.masks_from(data, flagged)
    prl = F.make_problem(data, C.KNOTS_FULL, ("lev",), masks=m)
    postl = F.fit(prl, h)
    res = {"test_id": TEST_IDS["r2"], "screening_full": {k: len(v) for k, v in flagged.items()}}
    pend = {}
    for fam in C.AFTER_FAMILIES:
        ex = _withhold(data, fam)
        mm = Hy.masks_from(data, flagged, ex)
        prh = F.make_problem(data, C.KNOTS_FULL, F.FAMILIES, masks=mm, fdict=fdict)
        posth = F.fit(prh, h)
        pend[fam] = _r2_targets(fam, data, m, ex, prh, posth, prl, postl, h, fdict)
        log(f"R2 {fam}: {len(pend[fam])} targets ({time.time() - t0:.0f} s)")
    reg = score_gate("r2")
    res["registration"] = reg
    for fam, T in pend.items():
        if not len(T):
            res[fam] = {"n": 0}
            continue
        T["obs"] = T.pop("_obs")
        T["err_fused"] = T.obs - T.pred_fused
        T["err_base"] = T.obs - T.pred_base
        S = T[T.vr >= C.VR_MIN]
        r = {"n_all": int(len(T)), "sites_all": int(T.site.nunique()), "n_scored": int(len(S)),
             "sites_scored": int(S.site.nunique()),
             "rmse_fused_all": rmse(T.err_fused), "rmse_base_all": rmse(T.err_base)}
        if len(S):
            r.update({"rmse_fused": rmse(S.err_fused), "rmse_base": rmse(S.err_base),
                      "ratio": rmse(S.err_fused) / max(rmse(S.err_base), 1e-12),
                      "bias_fused": float(S.err_fused.mean()),
                      "bias_base": float(S.err_base.mean())})
            if "sd_fused" in S and S.sd_fused.notna().all():
                r["coverage"] = coverage(np.abs(S.err_fused) <= Z90 * S.sd_fused, S.site)
        res[fam] = r
        T.to_parquet(C.EVIDENCE / f"r2_{fam}_targets.parquet", index=False)
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = None
    res["note"] = "after the fact (U7): report only, never a gate"
    F.dump(res, C.EVIDENCE / "r2.json")
    log("R2 done")
    return res


def _cm_present(pr, fam, month) -> bool:
    return f"gcm:{fam}:{month}" in pr.gindex


def _r2_targets(fam, data, m, ex, prh, posth, prl, postl, h, fdict) -> pd.DataFrame:
    rows = []
    if fam.startswith("gnss"):
        f = fam.split("_")[1]
        g = data["gnss"][m["gnss"]]
        g = g[g.family == f]
        for st, s in g.groupby("station"):
            pre = s[(s.t < 2023.0)]
            y22 = s[(s.t >= 2022.0) & (s.t < 2023.0)]
            post_ = s[s.t >= 2023.0]
            if len(y22) < 6 or len(post_) == 0 or len(pre[pre.t >= 2020.0]) < 12:
                continue
            xy = s[["x_km", "y_km"]].to_numpy()[0]
            anchor, t22 = float(y22.y.mean()), float(y22.t.mean())
            pr20 = pre[pre.t >= 2020.0]
            rate = ls_rate(pr20.t, pr20.y)
            t = post_.t.to_numpy()
            G = F.field_rows(prh, xy, t)
            gi = [prh.gindex[f"goff:{f}:{st}"], prh.gindex[f"grate:{f}:{st}"],
                  prh.gindex[f"gframe:{f}"]]
            G[:, prh.L + gi[0]] += 1.0
            G[:, prh.L + gi[1]] += t - C.T_REF
            G[:, prh.L + gi[2]] += t - C.T_REF
            extra = np.zeros(t.size)
            for j, mth in enumerate(post_.month):
                if _cm_present(prh, f, mth):
                    G[j, prh.L + prh.gindex[f"gcm:{f}:{mth}"]] += 1.0
                else:
                    extra[j] = h["sig_cm"] ** 2
            mu = G @ posth.theta
            var = posth.var_of(G) + extra + h["sig_gnss"] ** 2 + post_.sd.to_numpy() ** 2
            vh = field_increment_var(prh, posth, xy, t22, t)
            vl = field_increment_var(prl, postl, xy, t22, t)
            for j in range(t.size):
                rows.append({"site": st, "t": float(t[j]), "pred_fused": float(mu[j] - anchor),
                             "sd_fused": float(np.sqrt(var[j])),
                             "pred_base": rate * (float(t[j]) - t22),
                             "vr": float(1 - vh[j] / vl[j]),
                             "_obs": float(post_.y.iloc[j] - anchor)})
    elif fam in ("rings", "piles"):
        k = "ring" if fam == "rings" else "pile"
        c = data["cmp"][m["cmp"]]
        c = c[c.kind == k]
        for sen, s in c.groupby("sensor"):
            f = float(fdict.get(sen, 1.0))
            xy = s[["x_km", "y_km"]].to_numpy()[0]
            pre = s[s.t <= 2023.0 + 1e-6]
            if k == "ring":
                w = pre[(pre.t >= 2019.0)]
                if len(w) < 6:
                    continue
                base_rate = ls_rate(w.t, w.y) if w.seg.nunique() == 1 else ls_rate(
                    w[w.seg == w.seg.max()].t, w[w.seg == w.seg.max()].y)
            else:
                inc = []
                for _, sg in pre.groupby("seg"):
                    if len(sg) >= 2:
                        inc.append((sg.y.iloc[-1] - sg.y.iloc[0], sg.t.iloc[-1] - sg.t.iloc[0]))
                if not inc:
                    continue
                base_rate = sum(a for a, _ in inc) / sum(b for _, b in inc)
            for sg_id, _sg in s[s.t > 2023.0 + 1e-6].groupby("seg"):
                full_seg = s[s.seg == sg_id].sort_values("t")
                trn = full_seg[full_seg.t <= 2023.0 + 1e-6]
                anc = trn.iloc[-1] if len(trn) else full_seg.iloc[0]
                tgt = full_seg[full_seg.t > anc.t + 1e-6]
                tgt = tgt[tgt.t > 2023.0 + 1e-6]
                if not len(tgt):
                    continue
                t = tgt.t.to_numpy()
                ta = float(anc.t)
                G = f * (F.field_rows(prh, xy, t) - F.field_rows(prh, xy, np.full(t.size, ta)))
                mu = G @ posth.theta
                dt = t - ta
                var = posth.var_of(G) + dt ** 2 * h["sig_rc"] ** 2 + 2 * h["sig_cmp"] ** 2
                vh = field_increment_var(prh, posth, xy, ta, t)
                vl = field_increment_var(prl, postl, xy, ta, t)
                for j in range(t.size):
                    rows.append({"site": sen, "t": float(t[j]), "t_anchor": ta,
                                 "pred_fused": float(mu[j]), "sd_fused": float(np.sqrt(var[j])),
                                 "pred_base": base_rate * float(dt[j]),
                                 "vr": float(1 - vh[j] / vl[j]),
                                 "_obs": float(tgt.y.iloc[j] - anc.y)})
    elif fam == "insar":
        ins = data["insar"]
        F.set_ell(prh, h["ell_km"])
        F.set_ell(prl, h["ell_km"])
        Pc = prh.lat.weights(prh.cell_xy)
        series = {}
        for gr in ins["groups"]:
            if gr["t"] < 2023.0:
                continue
            for u, y in zip(gr["unit"], gr["y"], strict=True):
                series.setdefault(int(u), []).append((gr["t"], y, gr["month"]))
        Zh = posth.theta[:prh.L].reshape(prh.K, prh.m)
        Zl = postl.theta[:prl.L].reshape(prl.K, prl.m)
        recs = []
        for u, s in series.items():
            if len(s) < 3:
                continue
            s = sorted(s)
            phi = Pc[ins["unit_cells"][u]].mean(0)
            ta = s[0][0]
            t = np.array([a for a, _, _ in s[1:]])
            dy = np.array([b for _, b, _ in s[1:]]) - s[0][1]
            Ha = prh.tb.rows([ta])
            H = prh.tb.rows(t)
            muh = (H - Ha) @ (Zh @ phi) + posth.theta[prh.L] * (t - ta)
            mul = (H - Ha) @ (Zl @ phi) + postl.theta[prl.L] * (t - ta)
            gh = np.stack([np.outer(H[j] - Ha[0], phi).ravel() for j in range(t.size)])
            Gh = np.zeros((t.size, prh.N))
            Gh[:, :prh.L] = gh
            Gh[:, prh.L] = t - ta
            Gl = np.zeros((t.size, prl.N))
            Gl[:, :prl.L] = gh
            Gl[:, prl.L] = t - ta
            vh = posth.var_of(Gh)
            vl = postl.var_of(Gl)
            for j in range(t.size):
                recs.append({"site": f"unit{u}", "seg": int(ins["units"].loc[u, "seg"]),
                             "month": s[j + 1][2], "x": float(ins["units"].loc[u, "x_km"]),
                             "y": float(ins["units"].loc[u, "y_km"]), "t": float(t[j]),
                             "f": float(muh[j]), "b": float(mul[j]), "o": float(dy[j]),
                             "vr": float(1 - vh[j] / vl[j])})
        R = pd.DataFrame(recs)
        out = []
        for _, g in R.groupby(["seg", "month"]):
            if len(g) < 6:
                continue
            X = np.column_stack([np.ones(len(g)), g.x - g.x.mean(), g.y - g.y.mean()])
            P = _detrend_proj(X)
            gg = g.copy()
            gg["pred_fused"] = P @ g.f.to_numpy()
            gg["pred_base"] = P @ g.b.to_numpy()
            gg["_obs"] = P @ g.o.to_numpy()
            gg["sd_fused"] = np.nan
            out.append(gg.drop(columns=["f", "b", "o"]))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["t1", "t2", "t3", "r2"])
    a = ap.parse_args(argv)
    {"t1": t1, "t2": t2, "t3": t3, "r2": r2}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

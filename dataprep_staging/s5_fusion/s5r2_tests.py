"""WP S5 repair round 2: re-tests of the repaired model (S5.repair_r2.v1).

    tools/cpu_slot.sh python -I s5r2_tests.py t1f    FRESH temporal block (registered, gating):
                                                     pseudo-origin 2020-01, leveling 2018-19
                                                     withheld (never an S5 temporal target)
    tools/cpu_slot.sh python -I s5r2_tests.py t1     registered T1 design (2020-21 withheld),
                                                     SECOND LOOK (conditional)
    tools/cpu_slot.sh python -I s5r2_tests.py t2     registered T2 design (10 km blocks, no
                                                     buffer), SECOND LOOK
    tools/cpu_slot.sh python -I s5r2_tests.py t2b    T2 with plan 4.3's 10 km buffer (corrected
                                                     design), SECOND LOOK
    tools/cpu_slot.sh python -I s5r2_tests.py t3     leave one instrument family out, pre-origin
                                                     (report only, as registered in round 1)

Every comparison with a withheld value happens after ``_gate`` (assert_can_score for T1F; a
logged second-look record for the others). Thresholds are those of S5.subsidence_fusion.v1:
T1-type PASS iff >= 20 benchmarks scored at VR >= 0.5, RMSE_fused < RMSE_baseline and the
coverage3 gate passes (clusters = 10 km complete-linkage blocks); T2-type PASS iff the coverage3
gate passes and RMSE_fused < RMSE_IDW.

Outputs: results/dataprep/reports/S5_evidence/repair_r2/<test>.json and <test>_targets.parquet
(``--dry <dir>``: synthetic values, no registration, written to <dir>).
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
import s5data as D  # noqa: E402
import s5fit as F  # noqa: E402
import s5model as M  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5r2_data as RD  # noqa: E402
import s5r2_model as RM  # noqa: E402
import s5tests as Ts  # noqa: E402

Z90 = Ts.Z90
TEST_T1F = "S5.r2.T1F_leveling_2018_2019_withheld_fresh"
DRY: Path | None = None
OUT = K.EVID
SECOND_LOOK_UNITS = {
    "t1": ["s5.lev.t1_targets"] + ["s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP"],
    "t2": ["s5.lev.t2_blocks", "s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP"],
    "t2b": ["s5.lev.t2_blocks", "s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP"],
    "t3": ["s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP", "s5.lev.t2_blocks"],
}
SECOND_LOOK_SCOPE = {"t1": "leveling+gnss+insar+rings+piles:2012-01..2022-12",
                     "t2": "leveling+gnss+insar+rings+piles:2012-01..2022-12",
                     "t2b": "leveling+gnss+insar+rings+piles:2012-01..2022-12",
                     "t3": "leveling+gnss+insar+rings+piles:2012-01..2022-12"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _gate(key: str) -> dict:
    if DRY is not None:
        return {"dry_run": True}
    from harness import prereg3
    if key == "t1f":
        r = prereg3.assert_can_score(K.PREREG_ID, C.WP, test_id=TEST_T1F)
        return {"prereg": K.PREREG_ID, "test_id": TEST_T1F, "prereg_sha256": r["sha256"],
                "post_registration_looks": len(r["post_registration_looks"]),
                "post_registration_look_list": r["post_registration_looks"][:20],
                "unverifiable_prior_looks": len(r["unverifiable_prior_looks"]),
                "status": "registered fresh test"}
    rec = prereg3.record_look(
        SECOND_LOOK_UNITS[key], SECOND_LOOK_SCOPE[key], C.WP, "second_look",
        prereg_id=K.PREREG_ID,
        note=(f"S5 repair round 2: SECOND LOOK '{key}' declared in {K.PREREG_ID} (spent units "
              "of S5.subsidence_fusion.v1); logged after the predictions and immediately before "
              "the comparison with the withheld values; a pass is conditional (second look)."))
    return {"second_look_logged": rec["time"], "status": "second look (conditional)"}


def _out(name: str) -> Path:
    return (DRY if DRY is not None else OUT) / name


def _hyper(name: str) -> dict:
    base = DRY if DRY is not None else OUT
    return json.loads((base / name).read_text())


def _load(*a, **kw):
    return RD.load_all2(*a, **kw)


def score_t1_like(T: pd.DataFrame, res: dict) -> dict:
    T["err_fused"] = T.obs - T.pred_fused
    T["err_levonly"] = T.obs - T.pred_levonly
    T["err_baseline"] = T.obs - T.pred_baseline
    T["hit_fused"] = np.abs(T.err_fused) <= Z90 * T.sd_fused
    T["hit_levonly"] = np.abs(T.err_levonly) <= Z90 * T.sd_levonly
    S = T[T.vr >= C.VR_MIN]
    res.update({"n_targets_all": int(len(T)), "n_benchmarks_all": int(T.sid.nunique()),
                "n_targets_scored": int(len(S)), "n_benchmarks_scored": int(S.sid.nunique()),
                "vr_quantiles": {str(q): float(T.vr.quantile(q))
                                 for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(T) else {}})
    if res["n_benchmarks_scored"] >= C.T1_MIN_SCORED:
        z = S.err_fused / S.sd_fused
        res["scored"] = {
            "rmse_fused": Ts.rmse(S.err_fused), "rmse_baseline": Ts.rmse(S.err_baseline),
            "rmse_levonly": Ts.rmse(S.err_levonly),
            "ratio_fused_baseline": Ts.rmse(S.err_fused) / Ts.rmse(S.err_baseline),
            "boot_ratio": Ts.boot_ratio(S.err_fused, S.err_baseline, S.cluster),
            "bias_fused": float(S.err_fused.mean()), "bias_baseline": float(S.err_baseline.mean()),
            "rms_z_fused": float(np.sqrt(np.mean(z ** 2))),
            "median_sd_fused": float(S.sd_fused.median()),
            "coverage_fused": Ts.coverage(S.hit_fused, S.cluster),
            "coverage_levonly": Ts.coverage(S.hit_levonly, S.cluster),
            "by_year": {int(y): {"n": int(len(g)), "rmse_fused": Ts.rmse(g.err_fused),
                                 "rmse_baseline": Ts.rmse(g.err_baseline),
                                 "bias_fused": float(g.err_fused.mean())}
                        for y, g in S.groupby(np.floor(S.t_new).astype(int))}}
        sc = res["scored"]
        res["criteria"] = {"evaluable": True, "skill": bool(sc["ratio_fused_baseline"] < 1.0),
                           "coverage": bool(sc["coverage_fused"]["pass"])}
    else:
        res["criteria"] = {"evaluable": False, "skill": False, "coverage": False}
    res["pass"] = bool(all(res["criteria"].values()))
    if len(T):
        res["all_targets"] = {"rmse_fused": Ts.rmse(T.err_fused),
                              "rmse_baseline": Ts.rmse(T.err_baseline),
                              "rmse_levonly": Ts.rmse(T.err_levonly),
                              "coverage_fused": Ts.coverage(T.hit_fused, T.cluster)}
    return res


def _predict_t1_like(pr_f, post_f, pr_l, post_l, h, train_lev, targets, cl, base_win):
    rows = []
    for sid, nw in targets.groupby("sid"):
        tr = train_lev[train_lev.sid == sid].sort_values("t")
        if len(tr) < C.T1_BASE_MIN_SURVEYS or tr.t.max() - tr.t.min() < C.T1_BASE_MIN_SPAN_YR:
            continue
        trb = tr[(tr["date"].astype(str) >= base_win[0]) & (tr["date"].astype(str)
                                                            <= base_win[1])]
        xy = tr[["x_km", "y_km"]].to_numpy()[0]
        ta, ya = float(tr.t.iloc[-1]), float(tr.y.iloc[-1])
        tn = nw.t.to_numpy()
        mf, sf = RM.lev_pred_heldout(pr_f, post_f, h, xy, tr.t, tr.y, tr.camp, tn, nw.camp)
        ml, sl = RM.lev_pred_heldout(pr_l, post_l, h, xy, tr.t, tr.y, tr.camp, tn, nw.camp)
        vf = RM.field_increment_var(pr_f, post_f, xy, ta, tn)
        vl = RM.field_increment_var(pr_l, post_l, xy, ta, tn)
        rate = Ts.ls_rate(trb.t, trb.y)
        for j in range(len(nw)):
            rows.append({"sid": sid, "cluster": cl.get(sid, -1), "t_a": ta,
                         "t_new": float(tn[j]), "pred_fused": float(mf[j] - ya),
                         "sd_fused": float(sf[j]), "pred_levonly": float(ml[j] - ya),
                         "sd_levonly": float(sl[j]),
                         "pred_baseline": rate * (float(tn[j]) - ta),
                         "var_field_fused": float(vf[j]), "var_field_levonly": float(vl[j]),
                         "vr": float(1.0 - vf[j] / vl[j]),
                         "_obs": float(nw.y.iloc[j] - ya)})
    return pd.DataFrame(rows)


# ============================================================================ T1F (fresh)
def t1f() -> dict:
    t0 = time.time()
    hy = _hyper("hyper_t1f.json")
    h, fdict = hy["hyper"], hy["fdict"]
    data, qc = _load("pre", lev_end=K.T1F_TRAIN_END, other_end=K.T1F_OTHER_END,
                     ring_win=("2017-01-01", K.T1F_OTHER_END))
    m = RD.masks_from(data, hy["flagged"])
    lev = data["lev"]
    train_lev = lev[m["lev"]]
    pr_f = RM.make_problem2(data, K.KNOTS_T1F, F.FAMILIES, masks=m, fdict=fdict)
    post_f = RM.fit2(pr_f, h)
    pr_l = RM.make_problem2(data, K.KNOTS_T1F, ("lev",), masks={"lev": m["lev"]})
    post_l = RM.fit2(pr_l, h)
    log(f"T1F fits done ({time.time() - t0:.0f} s); N {pr_f.N}")
    # targets: dates and campaign ids now; values kept private until the gate
    full = D.leveling(data["grid"])[0]
    if DRY is not None:
        full = _dry_targets(full)
    full["camp"] = RM.campaign_ids(full, RD.towns())
    d = full["date"].astype(str)
    tg = full[(d >= K.T1F_WITHHELD[0]) & (d <= K.T1F_WITHHELD[1])].copy()
    cl = Ts.benchmark_clusters(train_lev)
    T = _predict_t1_like(pr_f, post_f, pr_l, post_l, h, train_lev, tg, cl, K.T1F_BASE_WIN)
    log(f"T1F {len(T)} targets at {T.sid.nunique() if len(T) else 0} benchmarks; "
        f"VR >= 0.5 at {int((T.vr >= C.VR_MIN).sum()) if len(T) else 0} targets")
    reg = _gate("t1f")
    T["obs"] = T.pop("_obs")
    res = {"test_id": TEST_T1F, "registration": reg, "hyper": h, "qc_inputs": qc["inputs"],
           "design": "pseudo-origin 2020-01; training leveling <= 2017-12; GNSS (v4), rings, "
                     "piles <= 2019-12; no InSAR; targets = 2018-2019 surveys"}
    score_t1_like(T, res)
    res["seconds"] = round(time.time() - t0, 1)
    T.to_parquet(_out("t1f_targets.parquet"), index=False)
    F.dump(res, _out("t1f.json"))
    log(f"T1F pass {res['pass']} {json.dumps(res['criteria'])}")
    return res


def _dry_targets(full: pd.DataFrame) -> pd.DataFrame:
    import s5r2_dryrun as DR
    full = full.copy()
    full["camp"] = RM.campaign_ids(full, RD.towns())
    return DR.synth({"lev": full, "gnss": _empty_gnss(), "insar": {"units": pd.DataFrame(
        columns=["x_km", "y_km"]), "groups": []}, "cmp": _empty_cmp()})["lev"]


def _empty_gnss():
    return pd.DataFrame(columns=["family", "station", "seg", "x_km", "y_km", "t", "y"])


def _empty_cmp():
    return pd.DataFrame(columns=["sensor", "seg", "x_km", "y_km", "t", "y"])


# ============================================================================ T1 (second look)
def t1() -> dict:
    t0 = time.time()
    base = _hyper("hyper_main.json")
    h1 = _hyper("hyper_t1.json")["hyper"]
    data, qc = _load("pre")
    m = RD.masks_from(data, base["flagged"])
    lev = data["lev"]
    d = lev["date"].astype(str).to_numpy()
    train = m["lev"] & (d <= C.T1_BASE_WIN[1])
    held = m["lev"] & (d >= C.T1_WITHHELD[0]) & (d <= C.T1_WITHHELD[1])
    masks = dict(m)
    masks["lev"] = train
    pr_c = RM.make_problem2(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=masks)
    post_c = RM.fit2(pr_c, h1)
    fd = F.estimate_f(pr_c, post_c, data["cmp"][m["cmp"]])
    fdict = {k: v["f"] for k, v in fd.items()}
    del post_c
    pr_f = RM.make_problem2(data, C.KNOTS_PRE, F.FAMILIES, masks=masks, fdict=fdict)
    post_f = RM.fit2(pr_f, h1)
    pr_l = RM.make_problem2(data, C.KNOTS_PRE, ("lev",), masks={"lev": train})
    post_l = RM.fit2(pr_l, h1)
    log(f"T1 fits done ({time.time() - t0:.0f} s)")
    cl = Ts.benchmark_clusters(lev[m["lev"]])
    T = _predict_t1_like(pr_f, post_f, pr_l, post_l, h1, lev[train], lev[held], cl,
                         C.T1_BASE_WIN)
    reg = _gate("t1")
    T["obs"] = T.pop("_obs")
    res = {"test": "registered T1 design (S5.T1_leveling_2020_2021_withheld), repaired model",
           "registration": reg, "hyper_t1": h1, "f_t1": fd, "qc_inputs": qc["inputs"]}
    score_t1_like(T, res)
    res["seconds"] = round(time.time() - t0, 1)
    T.to_parquet(_out("t1_targets.parquet"), index=False)
    F.dump(res, _out("t1.json"))
    log(f"T1 (second look) pass {res['pass']} {json.dumps(res['criteria'])}")
    return res


# ============================================================================ T2 / T2b
def _t2(buffer_km: float | None) -> dict:
    t0 = time.time()
    base = _hyper("hyper_main.json")
    h, fdict = base["hyper"], base["fdict"]
    data, qc = _load("pre")
    m = RD.masks_from(data, base["flagged"])
    lev = data["lev"][m["lev"]]
    cl = Ts.benchmark_clusters(lev)
    pr = RM.make_problem2(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    c_all = RM.total2(pr, h)
    bxy = lev.groupby("sid")[["x_km", "y_km"]].first()
    rates = {}
    for sid, g in lev.groupby("sid"):
        if len(g) >= C.T1_BASE_MIN_SURVEYS and g.t.max() - g.t.min() >= C.T1_BASE_MIN_SPAN_YR:
            rates[sid] = (Ts.ls_rate(g.t, g.y), g[["x_km", "y_km"]].to_numpy()[0])
    rs = np.array(list(rates.keys()))
    rxy = np.array([v[1] for v in rates.values()])
    rv = np.array([v[0] for v in rates.values()])
    rcl = np.array([cl[s] for s in rs])
    radius = C.T2_IDW_RADIUS_KM if buffer_km is None else K.T2B_IDW_RADIUS_KM
    rows = []
    blocks = sorted(set(cl.values()))
    for k, b in enumerate(blocks):
        sids = [s for s, c in cl.items() if c == b]
        if buffer_km is None:
            out_ids = sids
        else:
            dmin = M.dist(bxy.to_numpy(), bxy.loc[sids].to_numpy()).min(1)
            out_ids = bxy.index[dmin <= buffer_km].tolist()
        cb = RM.contribution2(pr, "lev", h, sids=out_ids)
        post = RM.fit2(pr, h, contrib=F.subtract(c_all, cb))
        nb_ok = ~np.isin(rs, out_ids) & (rcl != b)
        for sid in sids:
            g = lev[lev.sid == sid].sort_values("t")
            if len(g) < 2:
                continue
            xy = g[["x_km", "y_km"]].to_numpy()[0]
            t = g.t.to_numpy()
            cp = g.camp.to_numpy()
            G = (F.field_rows(pr, xy, t[1:]) - F.field_rows(pr, xy, t[:-1])
                 + RM.camp_cols(pr, cp[1:]) - RM.camp_cols(pr, cp[:-1]))
            mu = G @ post.theta
            dt = np.diff(t)
            var = post.var_of(G) + dt ** 2 * h["sig_rb"] ** 2 + 2 * h["sig_lev"] ** 2
            dd = np.sqrt(((rxy - xy) ** 2).sum(1))
            nb = nb_ok & (dd <= radius)
            if nb.sum() >= C.T2_IDW_MIN_NEIGHBOURS:
                w = 1.0 / np.maximum(dd[nb], 0.1) ** C.T2_IDW_POWER
                idw = float(np.sum(w * rv[nb]) / np.sum(w))
            else:
                idw = np.nan
            obs = np.diff(g.y.to_numpy())
            for j in range(len(dt)):
                rows.append({"sid": sid, "cluster": int(b), "t0": float(t[j]),
                             "t1": float(t[j + 1]), "pred_fused": float(mu[j]),
                             "sd_fused": float(np.sqrt(var[j])),
                             "pred_idw": idw * float(dt[j]) if np.isfinite(idw) else np.nan,
                             "n_left_out": len(out_ids), "_obs": float(obs[j])})
        if (k + 1) % 10 == 0:
            log(f"T2{'b' if buffer_km else ''} block {k + 1}/{len(blocks)} "
                f"({time.time() - t0:.0f} s)")
    T = pd.DataFrame(rows)
    key = "t2" if buffer_km is None else "t2b"
    reg = _gate(key)
    T["obs"] = T.pop("_obs")
    T["err_fused"] = T.obs - T.pred_fused
    T["err_idw"] = T.obs - T.pred_idw
    T["hit_fused"] = np.abs(T.err_fused) <= Z90 * T.sd_fused
    B = T[np.isfinite(T.pred_idw)]
    z = T.err_fused / T.sd_fused
    res = {"test": ("registered T2 design (S5.T2_leveling_block_out), repaired model"
                    if buffer_km is None else
                    "T2 with plan 4.3's 10 km buffer (corrected design), repaired model"),
           "registration": reg, "buffer_km": buffer_km, "idw_radius_km": radius,
           "n_blocks": len(blocks), "n_targets": int(len(T)), "n_benchmarks": int(T.sid.nunique()),
           "n_with_baseline": int(len(B)), "coverage_fused": Ts.coverage(T.hit_fused, T.cluster),
           "rmse_fused": Ts.rmse(B.err_fused), "rmse_idw": Ts.rmse(B.err_idw),
           "ratio_fused_idw": Ts.rmse(B.err_fused) / Ts.rmse(B.err_idw),
           "boot_ratio": Ts.boot_ratio(B.err_fused, B.err_idw, B.cluster),
           "rmse_fused_all": Ts.rmse(T.err_fused), "bias_fused": float(T.err_fused.mean()),
           "rms_z": float(np.sqrt(np.mean(z ** 2))),
           "median_sd_fused": float(T.sd_fused.median()), "qc_inputs": qc["inputs"]}
    res["criteria"] = {"coverage": bool(res["coverage_fused"]["pass"]),
                       "skill": bool(res["ratio_fused_idw"] < 1.0)}
    res["pass"] = bool(all(res["criteria"].values()))
    res["seconds"] = round(time.time() - t0, 1)
    T.to_parquet(_out(f"{key}_targets.parquet"), index=False)
    F.dump(res, _out(f"{key}.json"))
    log(f"{key} pass {res['pass']} {json.dumps(res['criteria'])}")
    return res


def t2() -> dict:
    return _t2(None)


def t2b() -> dict:
    return _t2(K.T2B_BUFFER_KM)


# ============================================================================ T3 (report only)
def t3() -> dict:
    t0 = time.time()
    base = _hyper("hyper_main.json")
    h, fdict = base["hyper"], base["fdict"]
    data, qc = _load("pre")
    m = RD.masks_from(data, base["flagged"])
    out = {"test": "leave one instrument family out, pre-origin (report only)", "families": {}}
    pend = {}
    # GNSS out
    pr = RM.make_problem2(data, C.KNOTS_PRE, ("lev", "insar", "cmp"), masks=m, fdict=fdict)
    post = RM.fit2(pr, h)
    rows = []
    gn = data["gnss"][m["gnss"]]
    for (fam, st, sg), g in gn.groupby(["family", "station", "seg"]):
        t = g.t.to_numpy()
        if t.size < 6:
            continue
        X = np.column_stack([np.ones(t.size), t - t.mean()])
        P = np.eye(t.size) - X @ np.linalg.pinv(X)
        G = F.field_rows(pr, g[["x_km", "y_km"]].to_numpy()[0], t)
        R = (np.diag(h["sig_cm"] ** 2 + h["sig_gnss"] ** 2 + g.sd.to_numpy() ** 2)
             + RM.expcov(t, h["sig_gc"], h["tau_g"]))
        Sg = post.cov_of(G) + R
        rows.append({"key": (fam, st, sg), "zhat": P @ (G @ post.theta),
                     "sd": np.sqrt(np.clip(np.diag(P @ Sg @ P.T), 1e-12, None)), "P": P,
                     "_y": g.y.to_numpy(), "t": t})
    pend["gnss"] = rows
    # rings and piles out
    pr = RM.make_problem2(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=m)
    post = RM.fit2(pr, h)
    rows = []
    cm = data["cmp"][m["cmp"]]
    for sen, g in cm.groupby("sensor"):
        t = g.t.to_numpy()
        segs, inv = np.unique(g.seg.to_numpy(), return_inverse=True)
        X = np.zeros((t.size, 2 * segs.size))
        for kk in range(segs.size):
            sel = inv == kk
            X[sel, kk] = 1.0
            X[sel, segs.size + kk] = t[sel] - t[sel].mean()
        P = np.eye(t.size) - X @ np.linalg.pinv(X)
        f = float(fdict.get(sen, 1.0))
        G = f * F.field_rows(pr, g[["x_km", "y_km"]].to_numpy()[0], t)
        R = h["sig_cmp"] ** 2 * np.eye(t.size) + RM.expcov(t, h["sig_cc"], h["tau_c"])
        Sg = post.cov_of(G) + R
        rows.append({"key": (sen,), "zhat": P @ (G @ post.theta),
                     "sd": np.sqrt(np.clip(np.diag(P @ Sg @ P.T), 1e-12, None)), "P": P,
                     "_y": g.y.to_numpy(), "t": t})
    pend["cmp"] = rows
    # InSAR out: per-unit rate relative to the segment median
    pr = RM.make_problem2(data, C.KNOTS_PRE, ("lev", "gnss", "cmp"), masks=m, fdict=fdict)
    post = RM.fit2(pr, h)
    prl = RM.make_problem2(data, C.KNOTS_PRE, ("lev",), masks={"lev": m["lev"]})
    postl = RM.fit2(prl, h)
    F.set_ell(pr, h["ell_km"])
    Pc = pr.lat.weights(pr.cell_xy)
    ins = data["insar"]
    series = {}
    for gr in ins["groups"]:
        for u, y in zip(gr["unit"], gr["y"], strict=True):
            series.setdefault(int(u), []).append((gr["t"], y))
    urows = []
    Zf = post.theta[:pr.L].reshape(pr.K, pr.m)
    Zl = postl.theta[:prl.L].reshape(prl.K, prl.m)
    for u, s in series.items():
        if len(s) < 6:
            continue
        t = np.array([a for a, _ in s])
        yv = np.array([b for _, b in s])
        phi = Pc[ins["unit_cells"][u]].mean(0)
        H = pr.tb.rows(t)
        urows.append({"unit": u, "seg": int(ins["units"].loc[u, "seg"]),
                      "rate_fused": Ts.ls_rate(t, H @ (Zf @ phi) + post.theta[pr.L] * (t - 2012)),
                      "rate_levonly": Ts.ls_rate(t, H @ (Zl @ phi)
                                                 + postl.theta[prl.L] * (t - 2012)),
                      "_rate_obs": Ts.ls_rate(t, yv)})
    U = pd.DataFrame(urows)
    reg = _gate("t3")
    out["registration"] = reg
    for fam in ("gnss", "cmp"):
        e_all, z_all, hit, clu = [], [], [], []
        for r in pend[fam]:
            zz = r["P"] @ r["_y"]
            e = zz - r["zhat"]
            e_all.append(e)
            z_all.append(zz)
            hit.append(np.abs(e) <= Z90 * r["sd"])
            clu += [str(r["key"][:2] if fam == "gnss" else r["key"][0])] * e.size
        e, zz, hit = np.concatenate(e_all), np.concatenate(z_all), np.concatenate(hit)
        out["families"][fam] = {"n": int(e.size), "sites": len(set(clu)),
                                "rmse_fused": Ts.rmse(e), "rmse_zero_baseline": Ts.rmse(zz),
                                "ratio": Ts.rmse(e) / Ts.rmse(zz),
                                "coverage": Ts.coverage(hit, np.array(clu))}
    U["rate_obs"] = U.pop("_rate_obs")
    for col in ("rate_obs", "rate_fused", "rate_levonly"):
        U[col + "_rel"] = U[col] - U.groupby("seg")[col].transform("median")
    out["families"]["insar"] = {
        "units": int(len(U)),
        "r_fused": float(np.corrcoef(U.rate_obs_rel, U.rate_fused_rel)[0, 1]),
        "r_levonly": float(np.corrcoef(U.rate_obs_rel, U.rate_levonly_rel)[0, 1]),
        "rmse_fused": Ts.rmse(U.rate_obs_rel - U.rate_fused_rel),
        "rmse_levonly": Ts.rmse(U.rate_obs_rel - U.rate_levonly_rel)}
    out["seconds"] = round(time.time() - t0, 1)
    F.dump(out, _out("t3.json"))
    log(f"T3 done {json.dumps(out['families'], default=float)[:600]}")
    return out


def main(argv=None) -> int:
    global DRY
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["t1f", "t1", "t2", "t2b", "t3"])
    ap.add_argument("--dry", default=None)
    a = ap.parse_args(argv)
    if a.dry:
        import s5r2_dryrun as DR
        DR.install()
        DRY = Path(a.dry)
    else:
        p = _out(f"{a.what}.json")
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    {"t1f": t1f, "t1": t1, "t2": t2, "t2b": t2b, "t3": t3}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

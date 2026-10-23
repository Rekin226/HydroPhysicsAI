"""WP S5 after-the-fact diagnostics for the report (NOT registered; report only).

Run after every registered test was scored (look logged first: ``s5looks2.py diagnostics``).
Nothing here feeds back into the product, a gate, a threshold or a hyperparameter.

1. Breakdown of the registered T1 and T2 targets by county (Changhua zones Z1-Z3, Yunlin, other)
   and, for T1, by withheld year.
2. Plan risk "partial-column operators trade off against the deep-compaction term; report
   posterior correlations": for each ring/pile sensor, on the pre-origin fit with every family
   (hyper_pre.json, f fixed at its estimate),
   (a) a Gauss-Newton (Laplace) joint posterior of (f_s, theta): sd of f_s if it were free, and
       its correlation with the field displacement at the sensor 2012-01-01 -> 2023-01-01;
   (b) the local regression y = segment offsets + f u + r (t - tbar) on the posterior-mean u of
       the fit WITHOUT rings and piles (the registered f estimator plus a rate term): corr(f, r)
       and f with the rate term.

    tools/cpu_slot.sh python -I s5diag.py       -> S5_evidence/diagnostics.json
"""
from __future__ import annotations

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
from scipy import linalg as sla  # noqa: E402

Z90 = 1.6448536269514722


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def county_of_cells() -> dict:
    units = json.loads(C.UNITS_V3.read_text())["units"]
    out = {}
    for u in units:
        if u["id"] in ("dt.zone.Z1", "dt.zone.Z2", "dt.zone.Z3"):
            for c in u["cells"]:
                out[int(c)] = "changhua_" + u["id"].split(".")[-1]
        elif u["id"] in ("dt.tercile.west", "dt.tercile.mid", "dt.tercile.east"):
            for c in u["cells"]:
                out[int(c)] = "yunlin"
    return out


def group_table(T: pd.DataFrame, err_cols: list, hit_col: str | None) -> dict:
    out = {}
    for gname, g in T.groupby("group"):
        r = {"n": int(len(g)), "benchmarks": int(g.sid.nunique())}
        for c in err_cols:
            r["rmse_" + c.replace("err_", "")] = rmse(g[c])
            r["bias_" + c.replace("err_", "")] = float(g[c].mean())
        if hit_col:
            r["coverage"] = float(g[hit_col].mean())
        out[gname] = r
    return out


def breakdowns(data: dict) -> dict:
    cc = county_of_cells()
    lev = data["lev"]
    sid_cell = lev.groupby("sid")["cell"].first()
    sid_grp = {s: cc.get(int(c), "other") for s, c in sid_cell.items()}
    out = {}
    t1 = C.EVIDENCE / "t1_targets.parquet"
    if t1.exists():
        T = pd.read_parquet(t1)
        T["group"] = T.sid.map(sid_grp)
        T["year"] = np.floor(T.t_new).astype(int)
        S = T[T.vr >= C.VR_MIN]
        out["t1_scored_by_group"] = group_table(S, ["err_fused", "err_baseline", "err_levonly"],
                                                "hit_fused")
        out["t1_all_by_group"] = group_table(T, ["err_fused", "err_baseline", "err_levonly"],
                                             "hit_fused")
        out["t1_scored_by_year"] = {int(y): {"n": int(len(g)), "rmse_fused": rmse(g.err_fused),
                                             "rmse_baseline": rmse(g.err_baseline),
                                             "coverage": float(g.hit_fused.mean())}
                                    for y, g in S.groupby("year")}
        out["t1_vr_by_group"] = {k: {"n_targets": int(len(g)),
                                     "share_vr_ge_0.5": float((g.vr >= C.VR_MIN).mean()),
                                     "vr_median": float(g.vr.median())}
                                 for k, g in T.groupby("group")}
        out["t1_sd_vs_error"] = {
            "scored_median_sd_fused": float(S.sd_fused.median()),
            "scored_rms_z_fused": float(np.sqrt(np.mean((S.err_fused / S.sd_fused) ** 2))),
            "scored_rms_z_levonly": float(np.sqrt(np.mean((S.err_levonly / S.sd_levonly) ** 2)))}
    t2 = C.EVIDENCE / "t2_targets.parquet"
    if t2.exists():
        T = pd.read_parquet(t2)
        T["group"] = T.sid.map(sid_grp)
        B = T[np.isfinite(T.pred_idw)]
        out["t2_by_group"] = group_table(B, ["err_fused", "err_idw"], "hit_fused")
        out["t2_rms_z"] = float(np.sqrt(np.mean((T.err_fused / T.sd_fused) ** 2)))
        dt = T.t1 - T.t0
        out["t2_by_interval"] = {
            k: {"n": int(sel.sum()), "rmse_fused": rmse(T.err_fused[sel]),
                "coverage": float(T.hit_fused[sel].mean())}
            for k, sel in (("dt<0.75y", dt < 0.75), ("0.75-1.5y", (dt >= 0.75) & (dt < 1.5)),
                           ("dt>=1.5y", dt >= 1.5))}
    r2 = {}
    for fam in C.AFTER_FAMILIES:
        p = C.EVIDENCE / f"r2_{fam}_targets.parquet"
        if not p.exists():
            continue
        T = pd.read_parquet(p)
        S = T[T.vr >= C.VR_MIN]
        per = {}
        for site, g in S.groupby("site"):
            per[str(site)] = {"n": int(len(g)), "rmse_fused": rmse(g.err_fused),
                              "rmse_base": rmse(g.err_base),
                              "bias_fused": float(g.err_fused.mean())}
        worst = sorted(per.items(), key=lambda kv: -kv[1]["rmse_fused"])
        rest = S[~S.site.isin([w[0] for w in worst[:1]])]
        r2[fam] = {"worst_sites": dict(worst[:3]),
                   "without_worst_site": {"rmse_fused": rmse(rest.err_fused),
                                          "rmse_base": rmse(rest.err_base)}}
        if fam.startswith("gnss"):
            r2[fam]["per_site"] = per
    out["r2_sites"] = r2
    return out


def f_tradeoff(data: dict, base: dict) -> dict:
    h, fdict = base["hyper"], base["fdict"]
    m = Hy.masks_from(data, base["flagged"])
    # (a) Laplace joint (f, theta) on the full pre-origin fit
    pr = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    post = F.fit(pr, h)
    F.set_ell(pr, h["ell_km"])
    th = post.theta
    res = {}
    for s in pr._cmp_sites:
        f = float(fdict.get(s["sensor"], 1.0))
        W, _, _ = M.site_local(s["t"], s["seg"], h["sig_cmp"], h["sig_rc"])
        G = F.field_rows(pr, s["xy"], s["t"])
        u = G @ th
        b = (f * G).T @ (W @ u)
        Pff = float(u @ W @ u)
        Pib = sla.cho_solve((post.chol, True), b, check_finite=False)
        var_f = 1.0 / max(Pff - float(b @ Pib), 1e-12)
        g = (F.field_rows(pr, s["xy"], np.array([2023.0]), seasonal=False)
             - F.field_rows(pr, s["xy"], np.array([2012.0]), seasonal=False))[0]
        Pig = sla.cho_solve((post.chol, True), g, check_finite=False)
        bg = float(b @ Pig)
        cov_fg = -var_f * bg
        var_g = float(g @ Pig) + bg ** 2 * var_f
        res[s["sensor"]] = {"f": f, "sd_f_free": float(np.sqrt(var_f)),
                            "u_2012_2023_mm": float(g @ th),
                            "sd_u_fixed_f": float(np.sqrt(float(g @ Pig))),
                            "sd_u_free_f": float(np.sqrt(var_g)),
                            "corr_f_u": float(cov_fg / np.sqrt(var_f * var_g))}
    del post
    # (b) local regression with a rate term on the fit without rings and piles
    pr_c = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=m)
    post_c = F.fit(pr_c, h)
    cm = data["cmp"][m["cmp"]]
    for sen, grp in cm.groupby("sensor"):
        t = grp.t.to_numpy()
        if t.size < 4 or sen not in res:
            continue
        Gc = F.field_rows(pr_c, grp[["x_km", "y_km"]].to_numpy()[0], t)
        uc = Gc @ post_c.theta
        segs, inv = np.unique(grp.seg.to_numpy(), return_inverse=True)
        X = np.zeros((t.size, segs.size + 2))
        X[np.arange(t.size), inv] = 1.0
        X[:, -2] = uc
        X[:, -1] = t - t.mean()
        beta = np.linalg.lstsq(X, grp.y.to_numpy(), rcond=None)[0]
        cov = np.linalg.pinv(X.T @ X)
        res[sen].update({"f_with_rate": float(beta[-2]), "rate_mm_yr": float(beta[-1]),
                         "corr_f_rate": float(cov[-2, -1] / np.sqrt(cov[-2, -2] * cov[-1, -1]))})
    vals = pd.DataFrame(res).T
    summ = {"n_sensors": int(len(vals)),
            "median_abs_corr_f_u": float(vals.corr_f_u.abs().median()),
            "max_abs_corr_f_u": float(vals.corr_f_u.abs().max()),
            "median_abs_corr_f_rate": float(vals.corr_f_rate.abs().median()),
            "max_abs_corr_f_rate": float(vals.corr_f_rate.abs().max()),
            "median_sd_f_free": float(vals.sd_f_free.median()),
            "median_sd_u_inflation": float((vals.sd_u_free_f / vals.sd_u_fixed_f).median())}
    return {"summary": summ, "sensors": res}


def main() -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    data, _ = Hy.load("pre")
    out = {"note": "after the fact (report only); not registered",
           "breakdowns": breakdowns(data)}
    log("breakdowns done")
    out["f_tradeoff"] = f_tradeoff(data, base)
    out["seconds"] = round(time.time() - t0, 1)
    F.dump(out, C.EVIDENCE / "diagnostics.json")
    log(f"diagnostics done in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

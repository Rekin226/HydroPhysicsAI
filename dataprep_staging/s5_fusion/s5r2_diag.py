"""WP S5 repair round 2: diagnosis of the failed gate (written BEFORE any method change).

Reads only spent evidence and data covered by the logged looks ``diagnose_lev`` and
``diagnose_inst`` (s5r2_looks.py). The leveling window 2018-01..2019-12 is never read: every
leveling table is filtered on load (``_lev_no_fresh``) and the T1 configuration's training
surveys end 2019-12 by design, so 2018-19 surveys enter the T1 fits as training data exactly as
in the registered test (no new comparison with them is made).

    tools/cpu_slot.sh python -I s5r2_diag.py t1_decomp     family decomposition of T1
    tools/cpu_slot.sh python -I s5r2_diag.py persistence   leveling rate-anomaly persistence
                                                            2012-2017 (design-time window)
    tools/cpu_slot.sh python -I s5r2_diag.py insar_runP     run P vs leveling, 2020-05..2021
    tools/cpu_slot.sh python -I s5r2_diag.py t2_split       T2 coverage by interval length
                                                            (increments outside 2018-19)

Outputs: results/dataprep/reports/S5_evidence/repair_r2/diag/<cmd>.json (+ parquet).
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
import s5tests as Ts  # noqa: E402

OUT = C.EVIDENCE / "repair_r2" / "diag"
FRESH = ("2018-01-01", "2019-12-31")
Z90 = Ts.Z90


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _lev_no_fresh(lev: pd.DataFrame) -> pd.DataFrame:
    d = lev["date"].astype(str)
    return lev[(d < FRESH[0]) | (d > FRESH[1])]


def rmse(e) -> float:
    return Ts.rmse(e)


# ============================================================================ T1 decomposition
def t1_decomp() -> dict:
    """The registered T1 configuration (hyper_t1.json, main screening, T1 f re-estimation)
    refitted with instrument subsets; predictions at the spent T1 targets."""
    t0 = time.time()
    base, data, m = Ts.base_setup()
    h1 = json.loads((C.EVIDENCE / "hyper_t1.json").read_text())["hyper"]
    lev = data["lev"]
    train = m["lev"] & (lev["date"].to_numpy() <= C.T1_BASE_WIN[1])
    masks = dict(m)
    masks["lev"] = train
    pr_c = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=masks)
    post_c = F.fit(pr_c, h1)
    fd = F.estimate_f(pr_c, post_c, data["cmp"][m["cmp"]])
    fdict = {k: v["f"] for k, v in fd.items()}
    subsets = {"all": F.FAMILIES, "lev_gnss": ("lev", "gnss"), "lev_insar": ("lev", "insar"),
               "lev_gnss_insar": ("lev", "gnss", "insar"), "lev_cmp": ("lev", "cmp"),
               "lev": ("lev",)}
    T = pd.read_parquet(C.EVIDENCE / "t1_targets.parquet")
    keys = T[["sid", "t_a"]].drop_duplicates()
    out = T[["sid", "cluster", "t_a", "t_new", "obs", "pred_baseline", "vr"]].copy()
    for name, fams in subsets.items():
        pr = F.make_problem(data, C.KNOTS_PRE, fams, masks=masks, fdict=fdict)
        post = F.fit(pr, h1)
        preds, sds, vfs = [], [], []
        for sid, _ta in keys.itertuples(index=False):
            tr = lev[train & (lev.sid == sid).to_numpy()]
            tn = T[T.sid == sid].t_new.to_numpy()
            xy = tr[["x_km", "y_km"]].to_numpy()[0]
            ya = float(tr.y.iloc[-1])
            mf, sf = Ts.pred_time_heldout(pr, post, h1, xy, tr.t, tr.y, tn)
            vf = Ts.field_increment_var(pr, post, xy, float(tr.t.iloc[-1]), tn)
            preds.append(pd.Series(mf - ya, index=T.index[T.sid == sid]))
            sds.append(pd.Series(sf, index=T.index[T.sid == sid]))
            vfs.append(pd.Series(vf, index=T.index[T.sid == sid]))
        out[f"pred_{name}"] = pd.concat(preds)
        out[f"sd_{name}"] = pd.concat(sds)
        out[f"vfield_{name}"] = pd.concat(vfs)
        log(f"{name}: done ({time.time() - t0:.0f} s)")
    S = out[out.vr >= C.VR_MIN].copy()
    S["yr"] = np.floor(S.t_new).astype(int)
    res = {"n_scored": int(len(S)), "subsets": {}}
    for name in subsets:
        e = S.obs - S[f"pred_{name}"]
        z = e / S[f"sd_{name}"]
        res["subsets"][name] = {
            "rmse": rmse(e), "bias": float(e.mean()), "rms_z": float(np.sqrt(np.mean(z ** 2))),
            "coverage_raw": float(np.mean(np.abs(z) <= Z90)),
            "median_sd": float(S[f"sd_{name}"].median()),
            "median_sd_field": float(np.sqrt(S[f"vfield_{name}"]).median()),
            "by_year": {int(y): {"rmse": rmse(g.obs - g[f"pred_{name}"]),
                                 "bias": float((g.obs - g[f"pred_{name}"]).mean())}
                        for y, g in S.groupby("yr")}}
    eb = S.obs - S.pred_baseline
    res["baseline"] = {"rmse": rmse(eb), "bias": float(eb.mean())}
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "t1_decomp.parquet", index=False)
    F.dump(res, OUT / "t1_decomp.json")
    log(json.dumps({k: {kk: round(v[kk], 2) for kk in ("rmse", "bias", "rms_z")}
                    for k, v in res["subsets"].items()}))
    return res


# ============================================================================ persistence
def persistence() -> dict:
    """Year-to-year persistence of benchmark rate anomalies, leveling 2012-2017 only."""
    import s5data as D
    from scipy.spatial import cKDTree
    g = D.grid()
    lev, _ = D.leveling(g)
    lev = _lev_no_fresh(lev)
    P = lev[lev["date"].astype(str) < FRESH[0]]
    rows = []
    for sid, gg in P.groupby("sid"):
        t, y = gg.t.to_numpy(), gg.y.to_numpy()
        if t.size < 5:
            continue
        dt = np.diff(t)
        r = np.diff(y) / dt
        ok = (dt > 0.7) & (dt < 1.3)
        if ok.sum() < 4:
            continue
        mr = (y[-1] - y[0]) / (t[-1] - t[0])
        for k in range(r.size):
            if ok[k]:
                rows.append({"sid": sid, "k": k, "yr": int(np.floor(t[k + 1])), "r": r[k],
                             "mr": mr, "x": gg.x_km.iloc[0], "y": gg.y_km.iloc[0],
                             "mon": int((t[k + 1] % 1) * 12) + 1})
    R = pd.DataFrame(rows)
    R["a"] = R.r - R.mr
    ye = R.groupby("yr").a.mean()
    R["a_loc"] = R.a - R.yr.map(ye)
    R = R.sort_values(["sid", "k"])
    for col in ("a", "a_loc"):
        R[col + "_next"] = R.groupby("sid")[col].shift(-1)
    R["k_next"] = R.groupby("sid").k.shift(-1)
    Q = R[R.k_next == R.k + 1]
    c_tot = float(np.corrcoef(Q.a, Q.a_next)[0, 1])
    c_loc = float(np.corrcoef(Q.a_loc, Q.a_loc_next)[0, 1])
    var_loc = float(R.a_loc.var())
    s2 = -c_loc * var_loc
    nb = []
    for _yr, q in R.groupby("yr"):
        tr = cKDTree(q[["x", "y"]].to_numpy())
        for i, xy in enumerate(q[["x", "y"]].to_numpy()):
            j = [k for k in tr.query_ball_point(xy, 8.0) if k != i]
            if len(j) >= 3:
                nb.append((q.a_loc.iloc[i], q.a_loc.iloc[j].mean()))
    nb = np.array(nb)
    res = {"window": "leveling 2012-01..2017-12 (consecutive increments 0.7-1.3 yr)",
           "benchmarks": int(R.sid.nunique()), "increments": int(len(R)),
           "year_mean_anomaly_mm_yr": {int(k): float(v) for k, v in ye.items()},
           "sd_anomaly_total": float(R.a.std()), "sd_year_common": float(ye.std()),
           "sd_anomaly_local": float(R.a_loc.std()),
           "lag1_corr_total": c_tot, "lag1_corr_local": c_loc, "lag1_pairs": int(len(Q)),
           "implied_displacement_noise_sd_mm": float(np.sqrt(max(s2, 0.0))),
           "implied_white_velocity_sd_mm_yr": float(np.sqrt(max(var_loc - 2 * s2, 0.0))),
           "corr_local_with_8km_neighbours": float(np.corrcoef(nb[:, 0], nb[:, 1])[0, 1]),
           "note": ("Under the round-1 IRW (velocity random walk) the lag-1 correlation of "
                    "annual-increment anomalies is >= 0 (persistent velocity changes) apart "
                    "from the survey white noise sig_lev; a negative value means displacement-"
                    "level anomalies that revert (e.g. head-driven recoverable compaction at the "
                    "survey date) or white velocity anomalies. The implied split assumes "
                    "anomaly = white displacement (sd s) + white velocity (sd v): corr = "
                    "-s^2 / (2 s^2 + v^2).")}
    OUT.mkdir(parents=True, exist_ok=True)
    F.dump(res, OUT / "persistence.json")
    log(json.dumps({k: res[k] for k in ("lag1_corr_total", "lag1_corr_local",
                                         "implied_displacement_noise_sd_mm",
                                         "implied_white_velocity_sd_mm_yr")}))
    return res


# ============================================================================ InSAR run P
def insar_runP() -> dict:
    """InSAR run-P cell change vs the leveling change at benchmarks surveyed in both 2020 and
    2021 (spent T1 window), with and without a fitted plane."""
    import s5data as D
    g = D.grid()
    z = np.load(C.WORK / "insar_runP_cells.npz", allow_pickle=False)
    val, flag = z["value"], z["flag"]
    dates = pd.to_datetime([str(d) for d in z["dates"]])
    lev, _ = D.leveling(g)
    lev = lev[lev["date"].astype(str) >= "2020-01-01"]
    cache = {}

    def at(d):
        if d not in cache:
            sel = np.abs((dates - d).days) <= 12
            v = np.where(flag[:, sel] == 0, val[:, sel], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cache[d] = np.nanmean(v, axis=1)
        return cache[d]

    rows = []
    for sid, gg in lev.groupby("sid"):
        if len(gg) < 2:
            continue
        a, b = gg.iloc[0], gg.iloc[-1]
        da, db = pd.Timestamp(a.date), pd.Timestamp(b.date)
        if (db - da).days < 300:
            continue
        c = int(a.cell)
        rows.append({"sid": sid, "da": a.date, "db": b.date, "lev": b.y - a.y,
                     "ins": -(at(db)[c] - at(da)[c]), "x": a.x_km, "y": a.y_km})
    R = pd.DataFrame(rows).dropna()
    res = {"pairs": {}}
    for (da, db), q in R.groupby([R.da.str[:7], R.db.str[:7]]):
        if len(q) < 15:
            continue
        X = np.column_stack([np.ones(len(q)), q.x - q.x.mean(), q.y - q.y.mean()])
        bl = np.linalg.lstsq(np.column_stack([X, q.ins]), q.lev, rcond=None)[0]
        rl = q.lev - X @ np.linalg.lstsq(X, q.lev, rcond=None)[0]
        ri = q.ins - X @ np.linalg.lstsq(X, q.ins, rcond=None)[0]
        res["pairs"][f"{da}->{db}"] = {
            "n": int(len(q)), "corr": float(np.corrcoef(q.lev, q.ins)[0, 1]),
            "corr_deplaned": float(np.corrcoef(rl, ri)[0, 1]),
            "slope_lev_on_ins_with_plane": float(bl[3]),
            "sd_lev_deplaned": float(rl.std()), "sd_ins_deplaned": float(ri.std()),
            "rms_diff_deplaned": float(np.sqrt(np.mean((rl - ri) ** 2))),
            "mean_lev": float(q.lev.mean()), "mean_ins": float(q.ins.mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_parquet(OUT / "insar_runP.parquet", index=False)
    F.dump(res, OUT / "insar_runP.json")
    log(json.dumps(res, default=float)[:800])
    return res


# ============================================================================ T2 split
def t2_split() -> dict:
    T = pd.read_parquet(C.EVIDENCE / "t2_targets.parquet")
    lo, hi = 2018.0, 2020.0
    T = T[((T.t0 < lo) & (T.t1 < lo)) | (T.t0 >= hi)]
    T["dt"] = T.t1 - T.t0
    T["z"] = T.err_fused / T.sd_fused
    res = {"n": int(len(T)), "note": "increments with both surveys outside 2018-2019"}
    for lab, q in T.groupby(pd.cut(T.dt, [0, 0.75, 1.5, 3, 20])):
        res[str(lab)] = {"n": int(len(q)), "coverage_raw": float(q.hit_fused.mean()),
                         "rms_z": float(np.sqrt(np.mean(q.z ** 2))),
                         "median_sd": float(q.sd_fused.median()), "rmse": rmse(q.err_fused)}
    F.dump(res, OUT / "t2_split.json")
    log(json.dumps(res, default=float)[:1200])
    return res


# ============================================================================ residual AC
def resid_ac() -> dict:
    """Lag-1 / lag-3 autocorrelation of the spent T3 residuals (GNSS, rings + piles)."""
    res = {}
    for name, f, key in (("cmp", "t3_cmp_targets.parquet", ["sensor"]),
                         ("gnss", "t3_gnss_targets.parquet", ["family", "station"])):
        R = pd.read_parquet(C.EVIDENCE / f)
        a1, a3 = [], []
        for _k, g in R.groupby(key):
            e = g.sort_values("t").err.to_numpy()
            if e.size < 12:
                continue
            e = e - e.mean()
            a1.append(float(np.sum(e[1:] * e[:-1]) / np.sum(e * e)))
            a3.append(float(np.sum(e[3:] * e[:-3]) / np.sum(e * e)))
        res[name] = {"sites": len(a1), "median_lag1": float(np.median(a1)),
                     "median_lag3": float(np.median(a3)), "rms_err": rmse(R.err)}
    hp = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    res["round1_noise"] = {k: hp["hyper"][k] for k in ("sig_cmp", "sig_rc", "sig_gnss",
                                                       "sig_ins", "ell_aps_km")}
    OUT.mkdir(parents=True, exist_ok=True)
    F.dump(res, OUT / "resid_ac.json")
    log(json.dumps(res))
    return res


# ============================================================================ GNSS v2 vs v4
def gnss_v2v4() -> dict:
    """gnss_clean/preorigin v2 vs v4 monthly tables as loaded by S5 (pre-origin)."""
    import s5data as D
    g = D.grid()
    a, _ = D.gnss(g, "pre")
    old = C.GNSS_PRE
    C.GNSS_PRE = C.DATAPREP / "gnss_clean" / "preorigin" / "v4"
    try:
        b, _ = D.gnss(g, "pre")
    finally:
        C.GNSS_PRE = old
    m = a.merge(b, on=["family", "station", "month"], how="outer", suffixes=("_2", "_4"),
                indicator=True)
    both = m[m._merge == "both"].copy()
    both["d"] = both.y_4 - both.y_2
    s = both.groupby(["family", "station"]).d.agg(["mean", "std", "size"])
    res = {"common_months": int(len(both)), "v4_only": int((m._merge == "right_only").sum()),
           "v2_only": int((m._merge == "left_only").sum()),
           "stations_changed": {f"{f}:{st}": {"mean_mm": float(r["mean"]),
                                              "sd_mm": float(r["std"]), "n": int(r["size"])}
                                for (f, st), r in s.iterrows() if r["std"] > 1.0
                                or abs(r["mean"]) > 1.0},
           "note": "y in mm, + = sinking (S5 sign); v4 - v2"}
    OUT.mkdir(parents=True, exist_ok=True)
    F.dump(res, OUT / "gnss_v2v4.json")
    log(json.dumps(res)[:800])
    return res


# ============================================================================ T1 second look
def t1sl_decomp() -> dict:
    """AFTER THE FACT (report only): the repaired T1 second-look configuration refitted with
    instrument subsets; predictions at its spent targets (nothing feeds back)."""
    import s5r2_data as RD
    import s5r2_model as RM
    import s5r2_tests as RT
    t0 = time.time()
    ev = C.EVIDENCE / "repair_r2"
    base = json.loads((ev / "hyper_main.json").read_text())
    h1 = json.loads((ev / "hyper_t1.json").read_text())["hyper"]
    T = pd.read_parquet(ev / "t1_targets.parquet")
    data, _ = RD.load_all2("pre")
    m = RD.masks_from(data, base["flagged"])
    lev = data["lev"]
    d = lev["date"].astype(str).to_numpy()
    train = m["lev"] & (d <= C.T1_BASE_WIN[1])
    masks = dict(m)
    masks["lev"] = train
    t1 = json.loads((ev / "t1.json").read_text())
    fdict = {k: v["f"] for k, v in t1["f_t1"].items()}
    res = {"note": "after the fact, report only; T1 second-look targets (spent)", "subsets": {}}
    S = T[T.vr >= C.VR_MIN]
    for name, fams in {"all": F.FAMILIES, "no_insar": ("lev", "gnss", "cmp"),
                       "no_cmp": ("lev", "gnss", "insar"), "no_gnss": ("lev", "insar", "cmp")
                       }.items():
        pr = RM.make_problem2(data, C.KNOTS_PRE, fams, masks=masks, fdict=fdict)
        post = RM.fit2(pr, h1)
        e, z = [], []
        for sid, nw in S.groupby("sid"):
            tr = lev[train & (lev.sid == sid).to_numpy()].sort_values("t")
            xy = tr[["x_km", "y_km"]].to_numpy()[0]
            camps = lev.loc[(lev.sid == sid).to_numpy(), ["t", "camp"]]
            cn = [camps.camp[np.isclose(camps.t, tt)].iloc[0] for tt in nw.t_new]
            mf, sf = RM.lev_pred_heldout(pr, post, h1, xy, tr.t, tr.y, tr.camp,
                                         nw.t_new.to_numpy(), cn)
            err = nw.obs.to_numpy() - (mf - float(tr.y.iloc[-1]))
            e.append(err)
            z.append(err / sf)
        e, z = np.concatenate(e), np.concatenate(z)
        res["subsets"][name] = {"rmse": rmse(e), "bias": float(e.mean()),
                                "rms_z": float(np.sqrt(np.mean(z ** 2))),
                                "coverage_raw": float(np.mean(np.abs(z) <= Z90))}
        log(f"{name}: {res['subsets'][name]} ({time.time() - t0:.0f} s)")
    res["baseline_rmse"] = rmse(S.obs - S.pred_baseline)
    F.dump(res, OUT / "t1sl_decomp.json")
    _ = RT
    return res


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["t1_decomp", "persistence", "insar_runP", "t2_split", "resid_ac",
                                       "gnss_v2v4", "t1sl_decomp"])
    a = ap.parse_args(argv)
    {"t1_decomp": t1_decomp, "persistence": persistence, "insar_runP": insar_runP,
     "t2_split": t2_split, "resid_ac": resid_ac, "gnss_v2v4": gnss_v2v4,
     "t1sl_decomp": t1sl_decomp}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

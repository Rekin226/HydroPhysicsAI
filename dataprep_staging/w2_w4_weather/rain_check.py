"""WP W4: rain residual gaps - masked IDW vs kriging with an elevation drift (KED),
leave-one-gauge-out (LOGO) on monthly gauge totals.

Implements pre-registration results/dataprep/preregistration/W4.rain_kriging_vs_idw.v1.json
(written before this script first ran):

* monthly totals by the twin's rule (calibrate_flow.monthly_mean_rate): a gauge-month is kept
  iff finite days >= 80 % of the month's days; total = mean of finite days x days in month;
* LOGO: each gauge g, each month where g is kept and >= 10 other gauges are kept;
* incumbent: hydrophysics.subsidence.idw_interp (power 2, eps 1e-6), i.e. the twin's masked
  IDW; candidate: KED with drift [1, altitude]; secondary: ordinary kriging (OK);
* variograms (exponential + nugget, on standardised OLS residuals pooled over months, 2.5 km
  lag bins on 0-40 km, WLS with pair-count weights) refitted per held-out gauge on the 25
  training gauges and the primary window only (2012-01..2022-12);
* decision on the primary window only: adopt KED iff RMSE_KED <= 0.90 RMSE_IDW fan-wide and
  RMSE_KED,g <= 1.20 RMSE_IDW,g for every gauge; the secondary window 2023-01..2026-09 is
  after-the-fact analysis.

Usage: python rain_check.py --out <product_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from wcommon import (CODIS_STATION_LIST, REPO, RF_STATIONS, RF_TIMESERIES, read_json_untrusted,
                     write_json_new)

sys.path.insert(0, str(REPO))
from hydrophysics.subsidence import idw_interp  # noqa: E402  (read-only import)

PRIMARY = ("2012-01", "2022-12")
SECONDARY = ("2023-01", "2026-09")
MIN_OTHERS = 10
LAG_BIN_KM = 2.5
MAX_LAG_KM = 40.0
SEED = 2026
N_BOOT = 2000
BLOCK = 12


# ----------------------------------------------------------------------------- data
def monthly_totals(path: Path = RF_TIMESERIES) -> pd.DataFrame:
    rf = pd.read_csv(path, parse_dates=["date time"]).set_index("date time")
    g = rf.resample("MS")
    mean, cnt = g.mean(), g.count()
    days = mean.index.days_in_month.to_numpy()[:, None]
    kept = cnt.to_numpy() >= 0.8 * days
    tot = mean * days
    return tot.where(kept)


def gauges() -> pd.DataFrame:
    st = pd.read_csv(RF_STATIONS, encoding="utf-8-sig")
    lst = read_json_untrusted(CODIS_STATION_LIST)
    alt = {}
    for grp in lst["data"]:
        for it in grp["item"]:
            alt[str(it["stationID"])] = float(it["altitude"])
    st["altitude_m"] = [alt[str(x)] for x in st["rf_num"]]
    return st.set_index("rf_id")[["rf_num", "TM_X97", "TM_Y97", "altitude_m"]]


# ----------------------------------------------------------------------------- variogram
def gamma(h_km: np.ndarray, c0: float, c1: float, a: float) -> np.ndarray:
    """Exponential variogram with nugget; gamma(0) = 0 exactly."""
    g = c0 + c1 * (1.0 - np.exp(-h_km / a))
    return np.where(h_km > 0, g, 0.0)


def experimental(tot: pd.DataFrame, gs: pd.DataFrame, cols: list[str], drift: bool) -> dict:
    """Pooled standardised experimental semivariogram over the primary window."""
    sub = tot.loc[PRIMARY[0]:PRIMARY[1], cols]
    xy = gs.loc[cols, ["TM_X97", "TM_Y97"]].to_numpy(float) / 1000.0
    z = gs.loc[cols, "altitude_m"].to_numpy(float)
    edges = np.arange(0.0, MAX_LAG_KM + LAG_BIN_KM, LAG_BIN_KM)
    nb = len(edges) - 1
    s_g = np.zeros(nb)
    s_h = np.zeros(nb)
    n = np.zeros(nb)
    used = 0
    for _, row in sub.iterrows():
        v = row.to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < MIN_OTHERS:
            continue
        y, p, zz = v[ok], xy[ok], z[ok]
        if drift:
            F = np.column_stack([np.ones_like(zz), zz])
            beta, *_ = np.linalg.lstsq(F, y, rcond=None)
            r = y - F @ beta
            dof = len(y) - 2
        else:
            r = y - y.mean()
            dof = len(y) - 1
        sd = np.sqrt((r ** 2).sum() / max(dof, 1))
        if not np.isfinite(sd) or sd < 1e-6:
            continue
        e = r / sd
        i, j = np.triu_indices(len(e), 1)
        h = np.hypot(*(p[i] - p[j]).T)
        sv = 0.5 * (e[i] - e[j]) ** 2
        k = np.digitize(h, edges) - 1
        m = (k >= 0) & (k < nb)
        np.add.at(s_g, k[m], sv[m])
        np.add.at(s_h, k[m], h[m])
        np.add.at(n, k[m], 1)
        used += 1
    keep = n > 0
    return {"h": (s_h[keep] / n[keep]), "g": (s_g[keep] / n[keep]), "n": n[keep],
            "months_used": used}


def fit_variogram(ex: dict) -> dict:
    h, g, n = ex["h"], ex["g"], ex["n"]
    w = np.sqrt(n)

    def res(p):
        return w * (gamma(h, *p) - g)

    sol = least_squares(res, x0=[0.2, 0.8, 10.0], bounds=([0, 0, 0.5], [5, 5, 200]))
    c0, c1, a = (float(x) for x in sol.x)
    return {"c0": c0, "c1": c1, "a_km": a, "nugget_ratio": c0 / (c0 + c1) if c0 + c1 > 0
            else 1.0, "months_used": int(ex["months_used"]), "n_pairs": int(n.sum()),
            "wls_cost": float(sol.cost), "bins": [[float(a1), float(b1), int(c)] for a1, b1, c
                                                  in zip(h, g, n)]}


# ----------------------------------------------------------------------------- kriging
def krige(p_tr: np.ndarray, y: np.ndarray, p0: np.ndarray, vg: dict,
          z_tr: np.ndarray | None = None, z0: float | None = None) -> tuple[float, bool]:
    """Universal kriging (drift [1, z] when z given, else ordinary). Returns (pred, fallback)."""
    n = len(y)
    d = np.hypot(*(p_tr[:, None, :] - p_tr[None, :, :]).transpose(2, 0, 1))
    G = gamma(d, vg["c0"], vg["c1"], vg["a_km"])
    g0 = gamma(np.hypot(*(p_tr - p0).T), vg["c0"], vg["c1"], vg["a_km"])
    if z_tr is None:
        F, f0 = np.ones((n, 1)), np.array([1.0])
    else:
        F, f0 = np.column_stack([np.ones(n), z_tr]), np.array([1.0, z0])
    k = F.shape[1]
    A = np.zeros((n + k, n + k))
    A[:n, :n], A[:n, n:], A[n:, :n] = G, F, F.T
    b = np.concatenate([g0, f0])
    fallback = False
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        fallback = True
    return float(sol[:n] @ y), fallback


# ----------------------------------------------------------------------------- LOGO
def run_logo(tot: pd.DataFrame, gs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = list(gs.index)
    tot = tot[cols]
    xy_m = gs[["TM_X97", "TM_Y97"]].to_numpy(float)
    xy_km = xy_m / 1000.0
    z = gs["altitude_m"].to_numpy(float)
    months = tot.loc[PRIMARY[0]:SECONDARY[1]].index
    vals = tot.loc[months].to_numpy(float)
    rows, vgs, n_fallback = [], {}, 0
    for gi, g in enumerate(cols):
        tr = [c for c in cols if c != g]
        tri = np.array([cols.index(c) for c in tr])
        vk = fit_variogram(experimental(tot, gs, tr, drift=True))
        vo = fit_variogram(experimental(tot, gs, tr, drift=False))
        vgs[g] = {"ked": vk, "ok": vo}
        for mi, mon in enumerate(months):
            obs = vals[mi, gi]
            if not np.isfinite(obs):
                continue
            yv = vals[mi, tri]
            ok = np.isfinite(yv)
            if ok.sum() < MIN_OTHERS:
                continue
            y = yv[ok]
            ptr_m, ptr_km, ztr = xy_m[tri][ok], xy_km[tri][ok], z[tri][ok]
            p_idw = float(idw_interp(xy_m[gi:gi + 1], ptr_m, y.reshape(-1, 1))[0, 0])
            p_ked, fb1 = krige(ptr_km, y, xy_km[gi], vk, ztr, z[gi])
            p_ok, fb2 = krige(ptr_km, y, xy_km[gi], vo)
            n_fallback += int(fb1) + int(fb2)
            ym = mon.strftime("%Y-%m")
            rows.append({"gauge": g, "month": ym,
                         "window": "primary" if ym <= PRIMARY[1] else "secondary",
                         "obs_mm": obs, "n_train": int(ok.sum()), "pred_idw": p_idw,
                         "pred_ked": max(p_ked, 0.0), "pred_ked_unclipped": p_ked,
                         "pred_ok": max(p_ok, 0.0)})
    vgs["_fallbacks"] = n_fallback
    # informational all-gauge fit (not used in LOGO)
    vgs["_all_gauges"] = {"ked": fit_variogram(experimental(tot, gs, cols, drift=True)),
                          "ok": fit_variogram(experimental(tot, gs, cols, drift=False))}
    return pd.DataFrame(rows), vgs


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def summarise(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    e = {m: df[f"pred_{m}"].to_numpy() - df["obs_mm"].to_numpy() for m in ("idw", "ked", "ok")}
    fan = {m: rmse(e[m]) for m in e}
    out = {"n_gauge_months": int(len(df)), "n_gauges": int(df["gauge"].nunique()),
           "n_months": int(df["month"].nunique()),
           "rmse_mm": fan, "bias_mm": {m: float(np.mean(e[m])) for m in e},
           "mae_mm": {m: float(np.mean(np.abs(e[m]))) for m in e},
           "ratio_ked_idw": fan["ked"] / fan["idw"], "ratio_ok_idw": fan["ok"] / fan["idw"],
           "obs_mean_mm": float(df["obs_mm"].mean())}
    mon = df["month"].str[5:7].astype(int)
    wet = mon.between(5, 10)
    out["season_rmse_mm"] = {
        s: {m: rmse(e[m][msk.to_numpy()]) for m in e}
        for s, msk in (("wet_May_Oct", wet), ("dry_Nov_Apr", ~wet))}
    per = []
    for g, d in df.groupby("gauge", sort=False):
        r = {"gauge": g, "n": len(d)}
        for m in ("idw", "ked", "ok"):
            r[f"rmse_{m}"] = rmse(d[f"pred_{m}"].to_numpy() - d["obs_mm"].to_numpy())
            r[f"bias_{m}"] = float(np.mean(d[f"pred_{m}"] - d["obs_mm"]))
        r["ratio_ked_idw"] = r["rmse_ked"] / r["rmse_idw"]
        r["ratio_ok_idw"] = r["rmse_ok"] / r["rmse_idw"]
        per.append(r)
    per = pd.DataFrame(per)
    out["max_gauge_ratio_ked_idw"] = float(per["ratio_ked_idw"].max())
    out["argmax_gauge_ratio_ked_idw"] = str(per.loc[per["ratio_ked_idw"].idxmax(), "gauge"])
    out["n_gauges_ked_better"] = int((per["ratio_ked_idw"] < 1).sum())
    return out, per


def block_bootstrap(df: pd.DataFrame) -> dict:
    """Paired moving-block bootstrap over months (12-month blocks) of the fan-wide RMSE
    ratios KED/IDW and OK/IDW."""
    rng = np.random.default_rng(SEED)
    months = sorted(df["month"].unique())
    sq = {m: df.assign(se=(df[f"pred_{m}"] - df["obs_mm"]) ** 2).groupby("month")["se"]
          .agg(["sum", "count"]).reindex(months) for m in ("idw", "ked", "ok")}
    T = len(months)
    nblk = int(np.ceil(T / BLOCK))
    starts_all = np.arange(T - BLOCK + 1)
    rk, ro = [], []
    for _ in range(N_BOOT):
        st = rng.choice(starts_all, nblk)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:T]
        tot = {m: (sq[m]["sum"].to_numpy()[idx].sum(), sq[m]["count"].to_numpy()[idx].sum())
               for m in sq}
        r = {m: np.sqrt(tot[m][0] / tot[m][1]) for m in tot}
        rk.append(r["ked"] / r["idw"])
        ro.append(r["ok"] / r["idw"])
    q = [5, 50, 95]
    return {"n_boot": N_BOOT, "block_months": BLOCK, "seed": SEED,
            "ratio_ked_idw_p5_p50_p95": [float(x) for x in np.percentile(rk, q)],
            "ratio_ok_idw_p5_p50_p95": [float(x) for x in np.percentile(ro, q)],
            "p_ked_ratio_le_0.90": float(np.mean(np.array(rk) <= 0.90))}


def residual_gaps(tot: pd.DataFrame) -> pd.DataFrame:
    k = tot.loc[PRIMARY[0]:SECONDARY[1]].notna()
    yr = k.groupby(k.index.year).mean()
    out = yr.T
    out.columns = [f"kept_frac_{c}" for c in out.columns]
    out.insert(0, "kept_months_2012_2022", k.loc[PRIMARY[0]:PRIMARY[1]].sum())
    out.insert(1, "kept_months_2023_2026_09", k.loc[SECONDARY[0]:SECONDARY[1]].sum())
    first = {c: (k.index[k[c].to_numpy()][0].strftime("%Y-%m") if k[c].any() else "")
             for c in k.columns}
    out.insert(2, "first_kept_month", pd.Series(first))
    out.index.name = "gauge"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    tot = monthly_totals()
    gs = gauges()
    df, vgs = run_logo(tot, gs)
    pri, sec = df[df["window"] == "primary"], df[df["window"] == "secondary"]
    s_pri, per_pri = summarise(pri)
    s_sec, per_sec = summarise(sec)
    boot = block_bootstrap(pri)
    adopt = (s_pri["ratio_ked_idw"] <= 0.90) and (s_pri["max_gauge_ratio_ked_idw"] <= 1.20)
    decision = {
        "rule": "adopt KED iff RMSE_KED <= 0.90 RMSE_IDW fan-wide AND every gauge "
                "RMSE_KED,g <= 1.20 RMSE_IDW,g (primary window 2012-01..2022-12)",
        "fan_ratio_ked_idw": s_pri["ratio_ked_idw"],
        "fan_criterion_pass": bool(s_pri["ratio_ked_idw"] <= 0.90),
        "max_gauge_ratio_ked_idw": s_pri["max_gauge_ratio_ked_idw"],
        "gauge_criterion_pass": bool(s_pri["max_gauge_ratio_ked_idw"] <= 1.20),
        "adopt_ked": bool(adopt),
        "outcome": "KED adopted" if adopt else "masked IDW stays (KED not adopted)"}
    # ---- write (primary outputs: fit window 2022-12; secondary: 2026-09)
    pri.to_csv(out / "logo_predictions_2012_2022.csv", index=False)
    per_pri.to_csv(out / "logo_per_gauge_2012_2022.csv", index=False)
    sec.to_csv(out / "logo_predictions_2023_2026.csv", index=False)
    per_sec.to_csv(out / "logo_per_gauge_2023_2026.csv", index=False)
    write_json_new(out / "variograms_2012_2022.json", vgs)
    gaps = residual_gaps(tot)
    gaps.insert(0, "altitude_m", gs["altitude_m"])
    gaps.to_csv(out / "residual_gaps_2012_2026.csv")
    write_json_new(out / "validation.json", {
        "preregistration": "results/dataprep/preregistration/W4.rain_kriging_vs_idw.v1.json",
        "window": list(PRIMARY), "summary": s_pri, "bootstrap": boot, "decision": decision,
        "kriging_fallback_solves": vgs["_fallbacks"]})
    write_json_new(out / "secondary_2023_2026.json", {
        "label": "after-the-fact analysis only; variograms fitted on 2012-2022; never used "
                 "for the decision", "window": list(SECONDARY), "summary": s_sec})
    print(json.dumps({"decision": decision, "primary": {k: s_pri[k] for k in (
        "n_gauge_months", "rmse_mm", "ratio_ked_idw", "ratio_ok_idw",
        "max_gauge_ratio_ked_idw", "argmax_gauge_ratio_ked_idw")}, "bootstrap": boot,
        "secondary_ratio_ked_idw": s_sec["ratio_ked_idw"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

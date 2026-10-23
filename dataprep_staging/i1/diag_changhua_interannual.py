"""After-the-fact diagnosis of Changhua's year-to-year head signal (2026-10-19, user request).

Seven ladder rungs (R1..R1e) changed the Changhua forcing; none moved the interannual
correlation of the out-of-fold head anomalies above about 0.52. Before any structural rung,
this asks whether that signal is REGIONAL (shared across wells, so a missing driver a model
could carry) or WELL-SPECIFIC (drift, datum steps, local pumping no fan-scale model resolves).

On the design-time out-of-fold predictions (``stage3_per_entry.npz``) of R0, R1c and R1e-pw,
per county (Changhua, Yunlin):

1. Interannual series per well: own-mean anomaly minus its month-of-year climatology, then a
   centred 12-month running mean (at least 8 finite months per window).
2. Coherence of the observed series and of the residual (observed minus predicted): share of
   the first eigenvalue of the pairwise-complete correlation matrix, and the median pairwise
   correlation by separation (0-5, 5-10, 10-20, >20 km) and by same/different layer.
3. Share of each well's residual interannual variance explained by the county's regional
   residual (median across the other wells; leave-one-out) and, separately, by one step or one
   linear trend (best single breakpoint, at least 24 months each side).
4. Correlation of the regional observed and residual series with candidate drivers'
   interannual anomalies over the county's cells: census electricity (R1c inputs), rain
   recharge, canal deliveries (W1 v4) and the P6 public-well abstraction.

Report only (U7): never a gate, a threshold or a choice of rung. Logs one look
(``after_the_fact``) on the design-time out-of-fold heads.

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/diag_changhua_interannual.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(REPO))

from harness import prereg3  # noqa: E402

RUNS = {"R0": "cwa", "R1c": "ccz", "R1e-pw": "cpw"}
COUNTIES = {"changhua": "彰化縣", "yunlin": "雲林縣"}
OUT = REPO / "results/dataprep/reports/I1_changhua_interannual"
STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
COUNTY_MAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"
SW = REPO / "results/dataprep/surface_water_v4/model_npz/v1/surface_water_v4_model.npz"
PW = REPO / "results/dataprep/public_wells/model_npz/v1/public_wells_model.npz"
MIN_MONTHS = 60          # wells with fewer finite months are left out of the coherence stats
DIST_BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]


def run_dir(suffix: str) -> Path:
    return REPO / f"results/twin_runs/stage3_datum_gate_{suffix}"


def interannual(x: np.ndarray, cal_month: np.ndarray) -> np.ndarray:
    """Own-mean anomaly minus month-of-year climatology, then a centred 12-month mean."""
    a = x - np.nanmean(x)
    clim = np.array([np.nanmean(a[cal_month == m]) if np.isfinite(a[cal_month == m]).any()
                     else np.nan for m in range(12)])
    d = a - clim[cal_month]
    out = np.full_like(d, np.nan)
    for t in range(len(d)):
        w = d[max(0, t - 6):t + 6]
        if np.isfinite(w).sum() >= 8:
            out[t] = np.nanmean(w)
    return out


def corr(a: np.ndarray, b: np.ndarray, min_n: int = 24) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def coherence(series: np.ndarray, xy: np.ndarray, layer: np.ndarray) -> dict:
    """First-eigenvalue share and median pairwise correlations by distance and layer."""
    n = len(series)
    C = np.full((n, n), np.nan)
    for i in range(n):
        C[i, i] = 1.0
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = corr(series[i], series[j])
    ok = np.isfinite(C).all(axis=1)
    Cm = C[np.ix_(ok, ok)]
    ev = np.sort(np.linalg.eigvalsh(Cm))[::-1] if Cm.size else np.array([np.nan])
    iu = np.triu_indices(n, 1)
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))[iu] / 1000.0
    c = C[iu]
    same = (layer[:, None] == layer[None, :])[iu]
    by_dist = {f"{lo:g}-{hi:g} km" if hi < 1e9 else f">{lo:g} km":
               {"median_r": float(np.nanmedian(c[(d >= lo) & (d < hi)]))
                if np.isfinite(c[(d >= lo) & (d < hi)]).any() else None,
                "n_pairs": int(np.isfinite(c[(d >= lo) & (d < hi)]).sum())}
               for lo, hi in DIST_BINS}
    return {"n_wells": int(n), "n_wells_in_eof": int(ok.sum()),
            "eof1_share": float(ev[0] / ev.sum()) if ev.size and np.isfinite(ev).all() else None,
            "eof2_share": float(ev[1] / ev.sum()) if ev.size > 1 else None,
            "median_pair_r": float(np.nanmedian(c)),
            "median_pair_r_same_layer": float(np.nanmedian(c[same])),
            "median_pair_r_other_layer": float(np.nanmedian(c[~same])),
            "by_distance": by_dist}


def step_or_trend_share(r: np.ndarray, min_seg: int = 24) -> tuple[float, float]:
    """Share of variance of ``r`` explained by its best single step and by a linear trend."""
    m = np.isfinite(r)
    y = r[m] - r[m].mean()
    v = float((y ** 2).sum())
    if len(y) < 2 * min_seg or v == 0:
        return float("nan"), float("nan")
    best = 0.0
    for k in range(min_seg, len(y) - min_seg):
        a, b = y[:k], y[k:]
        best = max(best, len(a) * a.mean() ** 2 + len(b) * b.mean() ** 2)
    t = np.arange(len(r))[m].astype(float)
    t -= t.mean()
    trend = float((t @ y) ** 2 / (t @ t)) if t @ t > 0 else 0.0
    return best / v, trend / v


def driver_series(cal_month: np.ndarray, flags_by_county: dict) -> dict:
    """County-mean interannual anomalies of the candidate drivers on the 131 scored months."""
    with open(run_dir(RUNS["R1c"]) / "inputs.local.pkl", "rb") as fh:
        inp = pickle.load(fh)
    rec_dates = pd.DatetimeIndex(inp.dates)
    scored = rec_dates[1:]                                  # obs columns: months 1..T-1
    E = np.asarray(sum(inp.E_by_class.values()), dtype=float)[:, 1:]
    R = np.asarray(inp.recharge_field, dtype=float)[:, 1:]
    out: dict = {}
    with np.load(SW) as z:
        sw_d = pd.DatetimeIndex(pd.to_datetime([str(d) for d in z["dates"]]))
        sw = np.asarray(z["sw_m_per_day"], dtype=float)[:, sw_d.get_indexer(scored)]
    with np.load(PW) as z:
        pw_d = pd.PeriodIndex([str(d) for d in z["dates"]], freq="M")
        pw = np.asarray(z["q_m3d"], dtype=float).sum(axis=0)[
            :, pw_d.get_indexer(scored.to_period("M"))]
    for cname, flags in flags_by_county.items():
        sel = np.asarray(flags, dtype=bool)
        out[cname] = {
            "census_energy": interannual(E[sel].sum(axis=0), cal_month),
            "rain_recharge": interannual(R[sel].mean(axis=0), cal_month),
            "canal_deliveries": interannual(sw[sel].mean(axis=0), cal_month),
            "public_wells": interannual(pw[sel].sum(axis=0), cal_month)}
    return out, scored


def main() -> None:
    st = pd.read_parquet(STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    with np.load(COUNTY_MAP) as z:
        ch_flags = np.asarray(z["changhua"], dtype=bool)
    # Yunlin cells = active cells off the Changhua map (the fan's other in-grid county share is
    # small; the driver means are reported for orientation only)
    flags = {"changhua": ch_flags, "yunlin": ~ch_flags}
    rec = {"schema": "i1-diag-changhua-interannual/1", "runs": RUNS,
           "role": "after_the_fact_analysis (U7: never a gate, threshold or choice)",
           "method": __doc__.split("\n\n")[2].strip(), "counties": {}}
    T = None
    drivers = scored = None
    for rname, suf in RUNS.items():
        z = np.load(run_dir(suf) / "stage3_per_entry.npz", allow_pickle=True)
        w = pd.read_csv(run_dir(suf) / "stage3_kfold_wells.csv", dtype={"sid": str}
                        ).set_index("entry")
        sids = w.loc[z["entry"], "sid"].to_numpy()
        county = st.loc[sids, "CountyName"].to_numpy()
        obs, pred = z["obs"].astype(float), z["pred"].astype(float)
        T = obs.shape[1]
        if drivers is None:
            # calendar month of each scored column (record starts 2012-01; columns = 2012-02..)
            drivers, scored = driver_series((np.arange(1, T + 1)) % 12, flags)
        cal = np.asarray(scored.month - 1)
        for cname, cc in COUNTIES.items():
            idx = [i for i in np.where(county == cc)[0]
                   if (np.isfinite(obs[i]) & np.isfinite(pred[i])).sum() >= MIN_MONTHS]
            o = np.array([interannual(np.where(np.isfinite(pred[i]), obs[i], np.nan), cal)
                          for i in idx])
            p = np.array([interannual(np.where(np.isfinite(obs[i]), pred[i], np.nan), cal)
                          for i in idx])
            res = o - p
            xy = np.c_[z["x"][idx], z["y"][idx]].astype(float)
            lay = np.asarray(z["layer"])[idx]
            block = rec["counties"].setdefault(cname, {})
            if "observed" not in block:
                block["observed"] = coherence(o, xy, lay)
                reg_o = np.nanmedian(o, axis=0)
                block["observed"]["regional_vs_drivers_r"] = {
                    k: corr(reg_o, v) for k, v in drivers[cname].items()}
                block["observed"]["per_layer_n"] = {str(int(k)): int((lay == k).sum())
                                                    for k in np.unique(lay)}
            r_blk = coherence(res, xy, lay)
            reg = np.nanmedian(res, axis=0)
            loo, steps, trends, pred_r = [], [], [], []
            for k in range(len(idx)):
                others = np.nanmedian(np.delete(res, k, axis=0), axis=0)
                rr = corr(res[k], others)
                loo.append(rr ** 2 if np.isfinite(rr) else np.nan)
                s, tr = step_or_trend_share(res[k])
                steps.append(s)
                trends.append(tr)
                pred_r.append(corr(o[k], p[k]))
            loo, steps, trends = map(np.asarray, (loo, steps, trends))
            r_blk.update({
                "interannual_r_obs_pred_median": float(np.nanmedian(pred_r)),
                "residual_var_share_regional_loo_median": float(np.nanmedian(loo)),
                "residual_var_share_one_step_median": float(np.nanmedian(steps)),
                "residual_var_share_trend_median": float(np.nanmedian(trends)),
                "wells_step_share_over_0p5": int(np.sum(steps > 0.5)),
                "wells_regional_share_over_0p5": int(np.sum(loo > 0.5)),
                "regional_residual_vs_drivers_r": {k: corr(reg, v)
                                                   for k, v in drivers[cname].items()},
                "regional_residual_sd_m": float(np.nanstd(reg)),
                "residual_sd_m_median": float(np.nanmedian(np.nanstd(res, axis=1)))})
            block.setdefault("residual", {})[rname] = r_blk
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        members={"I1:design_oof_heads_R0_R1e": {
            "window": "2012-01..2022-12",
            "description": "out-of-fold head predictions of the ladder rungs on R0's "
                           "design-time folds (stage3_per_entry.npz of R0, R1c, R1e-pw); "
                           "already scored by every rung's gates"}},
        note="Diagnosis of Changhua's interannual head signal (regional vs well-specific) "
             "after R1e; report only, never a gate or a rung choice. "
             "results/dataprep/reports/I1_changhua_interannual/report.json")
    print(json.dumps(rec["counties"], indent=1, ensure_ascii=False)[:12000])
    print("look logged:", look["time"])


if __name__ == "__main__":
    main()

"""Task 3: forward-chaining CV of the downscaling candidates (rules in ``gate_spec``).

Usage: python -I cv.py OUT_JSON [TPC_FILE ...]

Scores, per county (the in-grid census series the hybrid uses) and per 3 km disc:
* RMSE ratio to ``clim`` pooled over the test months of a fold set (pre 2018-2022,
  post 2023-2026H1, all), per test year, mean bias (%), and the anomaly skill
  1 - SSE/SSE_clim (the share of the month-to-month departure from climatology recovered);
* the correlation of predicted and true anomalies (both relative to the fold's ``clim``).
Discs: county prediction x the disc's month-of-year share (``model.shares``), against the
disc's own climatology and against the oracle (true county total x share).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_spec as G  # noqa: E402
import model as M  # noqa: E402
import panel  # noqa: E402

COUNTIES = ("changhua", "yunlin")
CLIM_START = "2012-01-01"   # the hybrid layer's driver start (its climatology window)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(tpc_files=None):
    tpc = panel.tpc_wide(tpc_files)
    nat = panel.national_ag()
    wx, _ = panel.weather()
    cen = panel.census_hybrid()
    keep = cen.county.index >= pd.Timestamp(CLIM_START)
    y = cen.county.where(cen.recorded, axis=0)[keep]
    yd = cen.discs.where(cen.recorded, axis=0)[keep]
    return tpc, nat, wx, cen, y, yd


def oof(tpc, nat, wx, y, yd, wc, models=M.MODELS):
    """Out-of-fold predictions: county {model: Series}, discs {model: DataFrame}."""
    pc = {c: {m: pd.Series(np.nan, index=y.index) for m in models} for c in COUNTIES}
    pdisc = {m: pd.DataFrame(np.nan, index=yd.index, columns=yd.columns)
             for m in list(models) + ["oracle", "disc_clim"]}
    for Y in G.TEST_YEARS:
        tr = y.index < pd.Timestamp(f"{Y}-01-01")
        te = (y.index.year == Y)
        idx = y.index[te]
        for c in COUNTIES:
            cols = wc.index[wc == c]
            sh = M.shares(yd.loc[tr, cols], y.loc[tr, c])
            shv = sh.reindex(idx.month).to_numpy()
            for m in models:
                f = M.fit(m, y.loc[tr, c], tpc, nat, wx, c)
                p = M.predict(f, idx, tpc, nat, wx)
                pc[c][m].loc[idx] = p.to_numpy()
                pdisc[m].loc[idx, cols] = p.to_numpy()[:, None] * shv
            pdisc["oracle"].loc[idx, cols] = y.loc[idx, c].to_numpy()[:, None] * shv
            dclim = yd.loc[tr, cols].groupby(yd.index[tr].month).mean()
            pdisc["disc_clim"].loc[idx, cols] = dclim.reindex(idx.month).to_numpy()
    return pc, pdisc


def scores(obs: pd.Series, pred: pd.Series, clim: pd.Series, years) -> dict:
    sel = obs.index.year.isin(years) & obs.notna() & pred.notna() & clim.notna()
    o, p, c = obs[sel], pred[sel], clim[sel]
    if len(o) == 0:
        return {}
    sse, ssc = float(((p - o) ** 2).sum()), float(((c - o) ** 2).sum())
    return {"n_months": int(len(o)), "rmse_GWh": round(float(np.sqrt(sse / len(o))), 4),
            "rmse_clim_GWh": round(float(np.sqrt(ssc / len(o))), 4),
            "rmse_ratio_vs_clim": round(float(np.sqrt(sse / ssc)), 4),
            "anomaly_skill": round(1.0 - sse / ssc, 4),
            "anomaly_corr": round(float(np.corrcoef(p - c, o - c)[0, 1]), 4),
            "bias_pct": round(float(100 * (p.sum() / o.sum() - 1)), 3),
            "mape_pct": round(float(100 * np.mean(np.abs(p / o - 1))), 3)}


def per_year(obs, pred, clim) -> dict:
    out = {}
    for Y in G.TEST_YEARS:
        s = scores(obs, pred, clim, (Y,))
        if s:
            out[str(Y)] = s["rmse_ratio_vs_clim"]
    return out


def coverage(obs, pred, sdlog, years) -> dict:
    sel = obs.index.year.isin(years) & obs.notna() & pred.notna()
    z = np.abs(np.log(obs[sel] / pred[sel])) / sdlog
    cov = float((z <= G.Z90).mean())
    n_eff = len(set(obs.index[sel].year))
    zz = 1.959963984540054
    den = 1 + zz ** 2 / n_eff
    cen = (cov + zz ** 2 / (2 * n_eff)) / den
    half = zz * np.sqrt(cov * (1 - cov) / n_eff + zz ** 2 / (4 * n_eff ** 2)) / den
    return {"coverage90": round(cov, 4), "n_months": int(sel.sum()), "n_eff_years": n_eff,
            "wilson95": [round(cen - half, 4), round(cen + half, 4)],
            "contains_0.90": bool(cen - half <= G.NOMINAL <= cen + half)}


def log_rms(obs, pred, years) -> float:
    sel = obs.index.year.isin(years) & obs.notna() & pred.notna()
    return float(np.sqrt(np.mean(np.log(obs[sel] / pred[sel]) ** 2)))


def disc_scores(yd, pdisc, wc, years) -> dict:
    out = {}
    for c in COUNTIES:
        cols = wc.index[wc == c]
        o = yd[cols]
        sel = o.index.year.isin(years)
        r = {}
        for m, P in pdisc.items():
            if m == "disc_clim":
                continue
            e = (P.loc[sel, cols] - o[sel]) ** 2
            ec = (pdisc["disc_clim"].loc[sel, cols] - o[sel]) ** 2
            ok = e.notna() & ec.notna()
            per = np.sqrt(e.where(ok).sum() / ec.where(ok).sum())
            r[m] = {"pooled_rmse_ratio_vs_disc_clim":
                    round(float(np.sqrt(e.where(ok).sum().sum() / ec.where(ok).sum().sum())), 4),
                    "median_well_ratio": round(float(per.median()), 4),
                    "share_wells_ratio_lt_1": round(float((per < 1).mean()), 4),
                    "n_wells": int(per.notna().sum())}
        out[c] = r
    return out


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    tpc_files = [Path(a) for a in argv[1:]] or None
    gate_hash = sha(HERE / "gate_spec.py")
    tpc, nat, wx, cen, y, yd = load(tpc_files)
    pc, pdisc = oof(tpc, nat, wx, y, yd, cen.well_county)
    res = {"gate_spec_sha256": gate_hash, "reads": cen.reads, "notes": cen.notes,
           "tpc_last_month": tpc.dropna(how="all").index.max().strftime("%Y-%m"),
           "county": {}, "selection": {}}
    sets = {"pre": G.PRE_YEARS, "post": G.POST_YEARS, "all": G.TEST_YEARS}
    for c in COUNTIES:
        clim = pc[c]["clim"]
        res["county"][c] = {m: {k: scores(y[c], pc[c][m], clim, v) for k, v in sets.items()}
                            | {"per_year_ratio": per_year(y[c], pc[c][m], clim)}
                            for m in M.MODELS}
    pre_mean = {m: float(np.mean([res["county"][c][m]["pre"]["rmse_ratio_vs_clim"]
                                  for c in COUNTIES])) for m in G.CANDIDATES}
    primary = min(pre_mean, key=pre_mean.get)
    res["selection"] = {"rule": "lowest mean pre-origin (2018-2022) RMSE ratio to clim",
                        "pre_mean_ratio": pre_mean, "primary": primary}
    cal = {}
    for c in COUNTIES:
        p = pc[c][primary]
        sd_pre = log_rms(y[c], p, G.PRE_YEARS)
        sd_all = log_rms(y[c], p, G.TEST_YEARS)
        cal[c] = {"sdlog_pre_folds": round(sd_pre, 4), "sdlog_all_folds": round(sd_all, 4),
                  "coverage_post_with_pre_sd": coverage(y[c], p, sd_pre, G.POST_YEARS),
                  "coverage_all_with_all_sd": coverage(y[c], p, sd_all, G.TEST_YEARS)}
        cr = {}
        for m in ("clim",):
            cr[m] = {"sdlog_pre": round(log_rms(y[c], pc[c][m], G.PRE_YEARS), 4)}
        cal[c]["clim_sdlog_pre"] = cr["clim"]["sdlog_pre"]
    res["sd_calibration"] = cal
    res["discs"] = {k: disc_scores(yd, pdisc, cen.well_county, v) for k, v in sets.items()}
    # disc sd: relative RMS of the primary's out-of-fold disc errors, per well and pooled
    P = pdisc[primary]
    rel = (yd - P) / P.where(P > 0)
    dsd = {}
    for c in COUNTIES:
        cols = cen.well_county.index[cen.well_county == c]
        r_all = rel.loc[rel.index.year.isin(G.TEST_YEARS), cols]
        r_pre = rel.loc[rel.index.year.isin(G.PRE_YEARS), cols]
        r_post = rel.loc[rel.index.year.isin(G.POST_YEARS), cols]
        pooled_pre = float(np.sqrt(np.nanmean(r_pre.to_numpy() ** 2)))
        per_pre = np.sqrt((r_pre ** 2).mean())
        z = (r_post.abs() / per_pre.fillna(pooled_pre)).to_numpy()
        dsd[c] = {"sdrel_pooled_all": round(float(np.sqrt(np.nanmean(r_all.to_numpy() ** 2))), 4),
                  "sdrel_pooled_pre": round(pooled_pre, 4),
                  "sdrel_per_well_all": {s: round(float(v), 4) for s, v in
                                         np.sqrt((r_all ** 2).mean()).items()},
                  "coverage90_post_with_pre_per_well_sd":
                      round(float(np.nanmean(z[np.isfinite(z)] <= G.Z90)), 4)}
    res["disc_sd"] = dsd
    g1 = {c: res["county"][c][primary]["pre"]["rmse_ratio_vs_clim"] <= G.G1_MAX_RATIO_PRE
          for c in COUNTIES}
    g2 = {c: res["county"][c][primary]["post"]["rmse_ratio_vs_clim"] < G.G2_MAX_RATIO_POST
          for c in COUNTIES}
    g3 = {c: cal[c]["coverage_post_with_pre_sd"]["contains_0.90"] for c in COUNTIES}
    res["gate_G1_G3"] = {"G1": g1, "G2": g2, "G3": g3,
                         "passed": all(g1.values()) and all(g2.values()) and all(g3.values())}
    oofp = pd.DataFrame({f"{c}_{m}": pc[c][m] for c in COUNTIES for m in M.MODELS})
    oofp.to_csv(out.with_suffix(".oof_county.csv"))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for c in COUNTIES:
        for m in M.MODELS:
            r = res["county"][c][m]
            print(c, m.ljust(9), "pre", r["pre"]["rmse_ratio_vs_clim"], "post",
                  r["post"]["rmse_ratio_vs_clim"], "skill pre/post", r["pre"]["anomaly_skill"],
                  r["post"]["anomaly_skill"], "corr", r["pre"]["anomaly_corr"],
                  r["post"]["anomaly_corr"], "bias post", r["post"]["bias_pct"])
    print("selection", res["selection"])
    print("sd", json.dumps(cal, indent=0))
    print("discs", json.dumps(res["discs"], indent=0))
    print("gate", res["gate_G1_G3"])


if __name__ == "__main__":
    main(sys.argv[1:])

"""WP P3 registered out-of-sample test: score the Yunlin-trained land-use / crop-statistics
maps, the scouts' published Changhua estimates and the heads-only inversion against the
Changhua census truth (WP C0 pump_electricity/preorigin/v3, Changhua county component,
2012-01..2022-12, opened through the loader guard as 'validator').

The Changhua census is read ONLY in ``main()`` after ``prereg3.assert_can_score`` succeeds
(pre-registration P3.changhua_transfer.v1). ``--dry-run`` replaces the truth with a synthetic
field built from the predictions themselves (no census value is read) to exercise the code.

Usage: python -I score.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crop_model as CM  # noqa: E402
import frame  # noqa: E402
import lib  # noqa: E402
import models_port as M  # noqa: E402

PREREG_ID = "P3.changhua_transfer.v1"
TOL_LOG = 0.30          # plan P4.3: |ln error| tolerance for a level (gate outcome B)
SHAPE = {"aa_delta_min": 0.03, "corr_min": 0.5}
R2_MIN = 0.5
TV_MAX = 0.10
BATTERY_SPEARMAN_MIN = 0.5
CLS_TRUTH = {"E_irrigation": ["irrigation"], "E_aquaculture": ["aquaculture"],
             "E_livestock": ["livestock"], "E_other": ["domestic", "industry", "other"]}
CLS_TRUTH_NODOM = {"E_irrigation": ["irrigation"], "E_aquaculture": ["aquaculture"],
                   "E_livestock": ["livestock"], "E_other": ["industry", "other"]}


# ----------------------------------------------------------------------------- fixed inputs
def phi_estimates(aux) -> dict:
    """In-grid fraction of Changhua county from non-census data (plan P3 step 7): MOI polygon
    area, LUIMAP farmland (paddy + dry + orchard) and MOA planted area (2012-22 mean planted
    ha-seasons per township x the township's in-grid area fraction)."""
    tz = aux["tz"]
    names = [str(n) for n in tz["names"]]
    groups = [str(g) for g in tz["groups"]]
    ch = [i for i, n in enumerate(names) if n.startswith("彰化縣")]
    infan_km2 = tz["share"].sum(1)
    area_km2 = tz["town_area_total_m2"] / 1e6
    fg = [groups.index(g) for g in ("paddy", "dry", "orchard")]
    farm_in = tz["town_class_area_m2"][:, fg].sum(1)
    farm_whole = np.nansum(tz["whole_class_area_m2"][:, fg], 1)
    t = lib.crop_table()
    planted = []
    for y in CM.YEARS:
        ct = t.loc[lib.roc(y)]
        planted.append(ct[["rice1", "rice2", "upland", "veg", "fruit", "special", "flower",
                           "forage"]].sum(1))
    pl = pd.concat(planted, axis=1).mean(1)
    frac = np.minimum(infan_km2 / area_km2, 1.0)
    w = np.array([pl.get(names[i], 0.0) for i in ch])
    return {"phi_area": float(infan_km2[ch].sum() / area_km2[ch].sum()),
            "phi_farm_luimap": float(farm_in[ch].sum() / farm_whole[ch].sum()),
            "phi_planted_moa": float((w * frac[ch]).sum() / w.sum()),
            "n_townships": len(ch)}


def published_estimates() -> dict:
    """The scouts' Changhua 'in fan' totals as published on 2026-10-12 (GWh/yr; before the
    census was obtained), with their domain convention."""
    def jl(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    ps = jl(lib.PILOT_LU / "prior_summary.json")
    cb = jl(lib.PILOT_LU / "combine.json")
    a2 = jl(lib.PILOT_ML / "a2_results.json")
    a4 = jl(lib.PILOT_ML / "a4_results.json")
    a5 = pd.read_csv(lib.PILOT_ML / "a5_scores.csv")
    e = jl(REPO_PILOT_INV / "e_changhua_inversion.json")
    out = {}
    for k, v in ps["changhua_total_GWh_by_model"].items():
        out[f"landuse_pilot|{k}"] = {"GWh": v, "family": "LUIMAP",
                                     "domain": "all cells x MOI Changhua share"}
    out["landuse_pilot|nnls_bootstrap_p50"] = {
        "GWh": ps["changhua_total_nnls_bootstrap_GWh"]["p50"], "family": "LUIMAP",
        "domain": "all cells x MOI Changhua share",
        "p05": ps["changhua_total_nnls_bootstrap_GWh"]["p05"],
        "p95": ps["changhua_total_nnls_bootstrap_GWh"]["p95"]}
    for k in ("2014-16", "2021"):
        out[f"landuse_pilot|crop_dasymetric_{k}"] = {
            "GWh": cb[k]["changhua_infan_total_GWh"], "family": "crop-statistics",
            "domain": "all cells x MOI Changhua share",
            "irrigation_GWh": cb[k]["changhua_infan_irrigation_GWh"]}
    for k, v in a2["changhua_totals"].items():
        if k.startswith("E_tot|"):
            out[f"design_ml_a2|{k[6:]}"] = {"GWh": v["changhua_GWh_yr"], "family": "LUIMAP",
                                            "domain": "cells with MOI Changhua share > 0.5"}
    for k, v in a4["changhua_GWh_yr"].items():
        if k.startswith("mlp"):
            out[f"design_ml_a4|{k}"] = {"GWh": v, "family": "LUIMAP",
                                        "domain": "cells with MOI Changhua share > 0.5"}
    for _, r in a5[a5.split == "CHANGHUA_GWh_yr"].iterrows():
        if r.model != "rf_total":
            out[f"design_ml_a5|{r.model}"] = {"GWh": float(r.ratio), "family": "LUIMAP",
                                              "domain": "cells with MOI Changhua share > 0.5"}
    out["heads_only|I2_smooth_noprior"] = {
        "GWh": e["I2_E_equiv_GWh_yr"], "family": "heads-only inversion (frozen theta)",
        "domain": "954 Changhua cells (TW_COUNTY label)"}
    return out


REPO_PILOT_INV = lib.REPO / "results/dataprep/_pilots/physics_inversion_hooks"


def heads_blocks(df) -> dict:
    """Reproduce the I2 inversion's 10 k-means blocks (inv_lib.make_blocks: KMeans k=10,
    n_init 10, random_state 0 on the 954 Changhua cell centroids in km) and check them
    against e_changhua_blocks_I2.csv."""
    from sklearn.cluster import KMeans
    d = np.load(REPO_PILOT_INV / "inputs_cwa.npz", allow_pickle=True)
    rows, cols = np.nonzero(d["mask"])
    cen = np.column_stack([float(d["x0"]) + (cols + 0.5) * float(d["dx"]),
                           float(d["y0"]) + (rows + 0.5) * float(d["dx"])])
    ch = np.asarray(d["cell_county"]).astype(str) == "彰化縣"
    idx = np.flatnonzero(ch)
    k = int(round(idx.size / 100))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(cen[idx] / 1000.0)
    lab = np.full(ch.size, -1)
    lab[idx] = km.labels_
    blk = pd.read_csv(REPO_PILOT_INV / "e_changhua_blocks_I2.csv")
    n_k = np.array([(lab == j).sum() for j in range(k)])
    xk = np.array([cen[lab == j, 0].mean() / 1000 for j in range(k)])
    yk = np.array([cen[lab == j, 1].mean() / 1000 for j in range(k)])
    ok = bool((n_k == blk.n_cells.to_numpy()).all() and
              np.allclose(xk, blk.x_km.to_numpy()) and np.allclose(yk, blk.y_km.to_numpy()))
    q = (blk.Q_irr_m3d + blk.Q_rest_m3d).to_numpy()
    return {"labels": lab, "q_m3d": q, "n_cells": n_k, "reproduced": ok, "k": k,
            "n_changhua_cells": int(ch.sum())}


# ----------------------------------------------------------------------------- truth
def load_truth(dry: bool, preds: dict):
    """Changhua truth: per-cell class kWh/yr, per-township (census town) tables, energy
    phi_in. ``dry``: synthetic stand-in built from the predictions (no census read)."""
    A = len(preds["moi_chg"])
    if dry:
        rng = np.random.default_rng(1)
        base = preds["cls"][list(preds["cls_names"]).index("lu_nnls_area")]
        chg = preds["moi_chg"]
        cells = {"irrigation": base[0] * chg, "aquaculture": base[1] * chg,
                 "livestock": base[2] * chg, "domestic": base[3] * chg * 0.5,
                 "industry": base[3] * chg * 0.2, "other": base[3] * chg * 0.3}
        cells = {k: v * rng.lognormal(0, 0.5, A) for k, v in cells.items()}
        tz = preds["tz"]
        names = [str(n) for n in tz["names"]]
        S = tz["share"]
        ch = [i for i, n in enumerate(names) if n.startswith("彰化縣")]
        town_ing = pd.DataFrame([{"town": names[j], "cls": c,
                                  "kwh": float((v * S[j] / np.maximum(chg, 1e-9)).sum())}
                                 for j in ch for c, v in cells.items()])
        years = []
        for y in CM.YEARS:
            for j in ch:
                years.append({"town": names[j], "year": y, "cls": "irrigation",
                              "kwh": float((cells["irrigation"] * S[j] /
                                            np.maximum(chg, 1e-9)).sum()) * 1.3})
        return cells, town_ing, pd.DataFrame(years), {"in_grid_GWh": 1.0, "all_GWh": 1.2}
    classes, E, dates, z = frame.class_cube("changhua", "validator")
    ny = len(dates) / 12
    cells = {c: E[classes.index(c)].sum(1) / ny for c in classes}
    tk = CM.town_kwh("changhua", "validator")
    town_ing = (tk.groupby(["town", "cls"])["kwh_ingrid"].sum() / len(CM.YEARS)).rename(
        "kwh").reset_index()
    town_year = tk.groupby(["town", "year", "cls"])["kwh"].sum().reset_index()
    tot_all = tk["kwh"].sum() / len(CM.YEARS) / 1e6
    tot_ing = tk["kwh_ingrid"].sum() / len(CM.YEARS) / 1e6
    return cells, town_ing, town_year, {"in_grid_GWh": tot_ing, "all_GWh": tot_all}


# ----------------------------------------------------------------------------- scoring
def r2(y, p):
    return M.r2(np.asarray(y, float), np.asarray(p, float))


def score_map(name, whole, chg, df, y, dom, S, ch_towns, town_truth):
    pred = whole * chg
    out = {"map": name, "pred_GWh": float(pred.sum() / 1e6)}
    for lab, dmask in dom.items():
        d = df[dmask]
        m = M.metrics(d, y[dmask], pred[dmask])
        out.update({f"{k}|{lab}": v for k, v in m.items()})
    pt = np.array([(whole * S[j]).sum() for j in ch_towns])
    tt = town_truth
    out["town_R2_level"] = r2(tt, pt)
    out["town_R2_share"] = r2(tt / tt.sum(), pt / max(pt.sum(), 1e-12))
    out["town_spearman"] = float(spearmanr(tt, pt)[0])
    return out


def class_split(name, cls_arr, chg, cells, mapping):
    pred = {c: float((cls_arr[i] * chg).sum()) for i, c in enumerate(M.CLS4)}
    tru = {c: float(sum(cells[k].sum() for k in mapping[c])) for c in M.CLS4}
    ps, ts = np.array([pred[c] for c in M.CLS4]), np.array([tru[c] for c in M.CLS4])
    tv = 0.5 * np.abs(ps / ps.sum() - ts / ts.sum()).sum()
    return {"map": name, "tv": float(tv),
            **{f"pred_share|{c}": float(ps[i] / ps.sum()) for i, c in enumerate(M.CLS4)},
            **{f"true_share|{c}": float(ts[i] / ts.sum()) for i, c in enumerate(M.CLS4)},
            **{f"ratio|{c}": float(ps[i] / ts[i]) if ts[i] > 0 else None
               for i, c in enumerate(M.CLS4)}}


def main(dry: bool) -> None:
    prereg = None
    if not dry:
        from harness import prereg3
        prereg = prereg3.assert_can_score(PREREG_ID, "P3")
    df, aux = frame.load("prior")
    z = np.load(lib.WORK / "predictions.npz", allow_pickle=True)
    preds = {"maps": z["maps"], "map_names": [str(x) for x in z["map_names"]],
             "cls": z["cls"], "cls_names": [str(x) for x in z["cls_names"]],
             "moi_chg": z["moi_chg"], "tz": aux["tz"], "prop": z["prop_changhua"],
             "gt05": z["changhua_gt05_cells"]}
    chg = preds["moi_chg"]
    cells, town_ing, town_year, tot = load_truth(dry, preds)
    y_tot = sum(cells.values())
    y_nodom = y_tot - cells["domestic"]
    tz = aux["tz"]
    names = [str(n) for n in tz["names"]]
    S = tz["share"]
    ch_towns = [i for i, n in enumerate(names) if n.startswith("彰化縣") and S[i].sum() >= 1.0]
    tw = town_ing.groupby("town")["kwh"].sum()
    tw_nodom = town_ing[town_ing.cls != "domestic"].groupby("town")["kwh"].sum()
    town_truth = np.array([tw.get(names[j], 0.0) for j in ch_towns])
    town_truth_nd = np.array([tw_nodom.get(names[j], 0.0) for j in ch_towns])
    dom = {"D": (chg > 0) | (y_tot > 0), "D05": chg > 0.5}
    truth = {"in_grid_GWh_all_classes": float(y_tot.sum() / 1e6),
             "in_grid_GWh_no_domestic": float(y_nodom.sum() / 1e6),
             "in_grid_GWh_cells_chg_gt05": float(y_tot[chg > 0.5].sum() / 1e6),
             "class_GWh": {k: float(v.sum() / 1e6) for k, v in cells.items()},
             "county_from_pumps": tot, "n_cells_D": int(dom["D"].sum()),
             "n_cells_D05": int(dom["D05"].sum()), "n_towns_infan": len(ch_towns),
             "census_towns_outside_infan_kwh": float(sum(v for t, v in tw.items()
                                                        if t not in [names[j]
                                                                     for j in ch_towns]))}
    T = truth["in_grid_GWh_all_classes"]
    TND = truth["in_grid_GWh_no_domestic"]
    # T2/T3/T4: rebuilt maps
    rows = []
    for k, name in enumerate(preds["map_names"]):
        r = score_map(name, preds["maps"][k], chg, df, y_tot, dom, S, ch_towns, town_truth)
        r2nd = score_map(name, preds["maps"][k], chg, df, y_nodom, {"D": dom["D"]}, S,
                         ch_towns, town_truth_nd)
        r.update({"ratio_all": r["pred_GWh"] / T, "ratio_nodom": r["pred_GWh"] / TND,
                  "AA_5km|D|nodom": r2nd["AA_5km|D"],
                  "corr_5km_density|D|nodom": r2nd["corr_5km_density|D"],
                  "town_R2_share|nodom": r2nd["town_R2_share"]})
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("map")
    uni = tab.loc["farm_uniform"]
    for lab in ("D", "D05"):
        tab[f"dAA_5km|{lab}"] = tab[f"AA_5km|{lab}"] - uni[f"AA_5km|{lab}"]
    tab["dAA_5km|D|nodom"] = tab["AA_5km|D|nodom"] - uni["AA_5km|D|nodom"]
    tab["level_pass"] = np.abs(np.log(tab["ratio_all"])) <= TOL_LOG
    tab["shape_pass"] = ((tab["dAA_5km|D"] >= SHAPE["aa_delta_min"]) &
                         (tab["corr_5km_density|D"] >= SHAPE["corr_min"]))
    tab["town_pass"] = tab["town_R2_share"] >= R2_MIN
    tab.to_csv(lib.EVID / ("scores_maps_dry.csv" if dry else "scores_maps.csv"))
    # T5: class split
    cs = []
    for i, name in enumerate(preds["cls_names"]):
        a = class_split(name, preds["cls"][i], chg, cells, CLS_TRUTH)
        b = class_split(name, preds["cls"][i], chg, cells, CLS_TRUTH_NODOM)
        a["tv_nodom"] = b["tv"]
        a["pass"] = a["tv"] <= TV_MAX
        cs.append(a)
    cs = pd.DataFrame(cs).set_index("map")
    # T1 / T6: published estimates (domain-matched truths too)
    pub = published_estimates()
    for v in pub.values():
        v["ratio_all"] = v["GWh"] / T
        v["ratio_nodom"] = v["GWh"] / TND
        v["ratio_domain_matched"] = v["GWh"] / (truth["in_grid_GWh_cells_chg_gt05"]
                                                if "> 0.5" in v["domain"] else T)
        v["pass_level"] = bool(abs(np.log(v["ratio_all"])) <= TOL_LOG)
    lu = [v["GWh"] for k, v in pub.items() if k.startswith("landuse_pilot|") and
          "bootstrap" not in k]
    rng_pub = [min(lu), max(lu)]
    # T4b: crop-statistics township model (whole townships, irrigation)
    cp = pd.read_csv(lib.WORK / "crop_model_changhua_pred.csv")
    irr_y = town_year[town_year.cls == "irrigation"]
    crop_rows = []
    for y in CM.YEARS + ["mean"]:
        if y == "mean":
            p = cp.groupby("town")["kwh_irr_whole_pred"].mean()
            t = irr_y.groupby("town")["kwh"].sum() / len(CM.YEARS)
        else:
            p = cp[cp.year == y].set_index("town")["kwh_irr_whole_pred"]
            t = irr_y[irr_y.year == y].groupby("town")["kwh"].sum()
        j = pd.concat([p.rename("pred"), t.rename("true")], axis=1).fillna(0.0)
        crop_rows.append({"year": y, "n_towns": len(j), "R2_level": r2(j.true, j.pred),
                          "R2_share": r2(j.true / j.true.sum(), j.pred / j.pred.sum()),
                          "spearman": float(spearmanr(j.true, j.pred)[0]),
                          "ratio": float(j.pred.sum() / j.true.sum()),
                          "true_GWh": float(j.true.sum() / 1e6),
                          "pred_GWh": float(j.pred.sum() / 1e6)})
    crop_tab = pd.DataFrame(crop_rows)
    # pilot variant (2014/15/16/21 betas, no shrinkage)
    with open(lib.EVID / "crop_model.json", encoding="utf-8") as f:
        cmj = json.load(f)
    pil_rows = []
    for y, v in cmj["pilot_variant"].items():
        p = pd.Series(v["changhua_whole_pred"])
        t = irr_y[irr_y.year == int(y)].groupby("town")["kwh"].sum()
        j = pd.concat([p.rename("pred"), t.rename("true")], axis=1).fillna(0.0)
        pil_rows.append({"year": int(y), "R2_level": r2(j.true, j.pred),
                         "R2_share": r2(j.true / j.true.sum(), j.pred / j.pred.sum()),
                         "ratio": float(j.pred.sum() / j.true.sum())})
    # one-rate baseline (Yunlin kWh per planted ha-season, per year)
    t_crop = lib.crop_table()
    yk = pd.read_csv(lib.WORK / "yunlin_town_kwh.csv")
    one_rows = []
    for y in CM.YEARS:
        Xy = CM.design(t_crop, y, "雲林縣")
        Yy = yk[(yk.cls == "irrigation") & (yk.year == y)].groupby("town")["kwh"].sum()
        jj = Xy.join(Yy.rename("kwh"), how="inner")
        rate = jj["kwh"].sum() / jj[CM.FEATS].to_numpy().sum()
        Xc = CM.design(t_crop, y, "彰化縣")
        p = Xc[CM.FEATS].sum(1) * rate
        t = irr_y[irr_y.year == y].groupby("town")["kwh"].sum()
        j = pd.concat([p.rename("pred"), t.rename("true")], axis=1).fillna(0.0)
        one_rows.append({"year": y, "R2_level": r2(j.true, j.pred), "rate_kwh_per_ha": rate,
                         "ratio": float(j.pred.sum() / j.true.sum())})
    # T6: heads-only block pattern
    hb = heads_blocks(df)
    lab = hb["labels"]
    tb = np.array([y_tot[lab == j].sum() for j in range(hb["k"])])
    q = hb["q_m3d"]
    nk = hb["n_cells"]
    heads = {"blocks_reproduced": hb["reproduced"], "k": hb["k"],
             "truth_GWh_on_954_cells": float(y_tot[lab >= 0].sum() / 1e6),
             "spearman_q_vs_truth": float(spearmanr(q, tb)[0]),
             "AA_blocks": float(1 - 0.5 * np.abs(q / q.sum() - tb / tb.sum()).sum()),
             "AA_blocks_uniform_by_cells": float(1 - 0.5 * np.abs(nk / nk.sum() -
                                                                 tb / tb.sum()).sum()),
             "n_blocks_zero_estimate": int((q < 1e-3 * q.max()).sum()),
             "truth_share_in_zero_blocks": float(tb[q < 1e-3 * q.max()].sum() / tb.sum())}
    # T7: phi_in
    phi = phi_estimates(aux)
    lo, hi = min(phi["phi_area"], phi["phi_farm_luimap"], phi["phi_planted_moa"]), \
        max(phi["phi_area"], phi["phi_farm_luimap"], phi["phi_planted_moa"])
    phi_census = tot["in_grid_GWh"] / tot["all_GWh"]
    # T8: battery vs Changhua
    with open(lib.EVID / "battery.json", encoding="utf-8") as f:
        bat = json.load(f)
    common = [m for m in bat["gate"] if m in tab.index and m != "farm_uniform"]
    bx = np.array([bat["gate"][m]["aa_mean_delta"] for m in common])
    cx = np.array([tab.loc[m, "dAA_5km|D"] for m in common])
    primary = bat["primary_by_plan_rule"]
    res = {
        "dry_run": dry, "prereg": None if prereg is None else
        {"id": PREREG_ID, "sha256": prereg["sha256"],
         "post_registration_looks": prereg["post_registration_looks"],
         "unverifiable_prior_looks": len(prereg["unverifiable_prior_looks"])},
        "truth": truth,
        "T1_level_scouts": {"estimates": pub, "published_range_GWh": rng_pub,
                            "truth_in_range": bool(rng_pub[0] <= T <= rng_pub[1]),
                            "truth_nodom_in_range": bool(rng_pub[0] <= TND <= rng_pub[1]),
                            "n_pass_level": int(sum(v["pass_level"] for v in pub.values())),
                            "n_estimates": len(pub)},
        "T2_level_rebuilt": {"n_pass": int(tab["level_pass"].sum()), "n_maps": len(tab),
                             "ratio_range": [float(tab["ratio_all"].min()),
                                             float(tab["ratio_all"].max())]},
        "T3_shape_5km": {"primary": primary,
                         "primary_row": tab.loc[primary, ["dAA_5km|D", "corr_5km_density|D",
                                                          "AA_5km|D", "shape_pass"]]
                         .to_dict(),
                         "n_pass": int(tab["shape_pass"].sum()),
                         "uniform_AA_5km": float(uni["AA_5km|D"]),
                         "uniform_corr_5km": float(uni["corr_5km_density|D"])},
        "T4_township": {"n_pass_share_R2": int(tab["town_pass"].sum()),
                        "crop_model": crop_tab.to_dict(orient="records"),
                        "crop_model_mean_pass": bool(crop_tab.iloc[-1]["R2_level"] >= R2_MIN),
                        "crop_model_pilot_variant": pil_rows, "one_rate": one_rows},
        "T5_class_split": cs.reset_index().to_dict(orient="records"),
        "T6_heads_only": {**heads, "ratio_all": pub["heads_only|I2_smooth_noprior"]["ratio_all"],
                          "pass_level": pub["heads_only|I2_smooth_noprior"]["pass_level"]},
        "T7_phi_in": {**phi, "registered_interval": [lo, hi], "census_energy_phi": phi_census,
                      "pass": bool(lo <= phi_census <= hi)},
        "T8_battery_predicts": {"maps": common, "battery_mean_dAA": bx.tolist(),
                                "changhua_dAA": cx.tolist(),
                                "spearman": float(spearmanr(bx, cx)[0]),
                                "pass": bool(spearmanr(bx, cx)[0] >= BATTERY_SPEARMAN_MIN),
                                "primary_battery": bat["gate"][primary]},
    }
    # shift: error of the primary map by propensity (cells with MOI share > 0.5)
    gt = preds["gt05"]
    pr = preds["prop"]
    k = preds["map_names"].index(primary)
    pp = preds["maps"][k][gt] * chg[gt]
    yy = y_tot[gt]
    hi_p = pr > 0.9
    res["shift_vs_error"] = {
        "frac_cells_prop_gt_0.9": float(hi_p.mean()),
        "ratio_prop_gt_0.9": float(pp[hi_p].sum() / max(yy[hi_p].sum(), 1e-9)),
        "ratio_prop_le_0.9": float(pp[~hi_p].sum() / max(yy[~hi_p].sum(), 1e-9))}
    out = lib.EVID / ("scores_dry.json" if dry else "scores.json")
    lib.save_json(out, res)
    cs.to_csv(lib.EVID / ("scores_class_dry.csv" if dry else "scores_class.csv"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("T5_class_split",)},
                     ensure_ascii=False, indent=1, default=lib._jdefault)[:8000])


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
